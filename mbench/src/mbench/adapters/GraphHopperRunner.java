///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 17+
//DEPS com.graphhopper:graphhopper-map-matching:10.2

// GraphHopper map-matching runner for the mbench GraphHopper baseline.
//
// Runs entirely in-process on the embeddable JVM library (no GraphHopper server).
// jbang resolves graphhopper-map-matching (pinned above) and its transitive deps
// from Maven Central and caches them; the Python adapter shells out to
// `jbang GraphHopperRunner.java ...`.
//
// Formulation (mirrors the Valhalla Meili adapter): each local target segment is a
// densified synthetic GPS trace; we map-match it onto the Overture-derived routable
// graph (built once by mbench/convert/pbf.py) and read back the matched reference
// edges. GraphHopper does not expose OSM way ids on matched edges, so convert/pbf.py
// writes the synthetic way_id into each way's `name` tag (KVStorage) and we recover
// it here via edge.getName(). Aggregation + the overlap threshold live on the Python
// side (mapmatch_common.aggregate_edges), shared with Meili.
//
// I/O is line-oriented TSV (no JSON dependency):
//   traces  in : "<target_id>\t<lon>,<lat>;<lon>,<lat>;..."   (one trace per line)
//   matches out: "<target_id>\t<way_id>,<edge_m>,<n_states>;..."
//                (per matched edge: full edge length in m + number of trace points
//                 snapped onto it; summed per way_id. n_states lets the Python side
//                 drop "bridged" edges — routed between observations with no point
//                 actually on them — which are GraphHopper's parallel/connecting
//                 false positives and have no analogue in Valhalla's map_snap.)
//
// Args: <pbf> <gh_graph_dir> <traces_tsv> <out_tsv> <vehicle> <measurement_sigma_m> <min_network_size> <workers>

import static com.graphhopper.json.Statement.If;
import static com.graphhopper.json.Statement.Op.LIMIT;
import static com.graphhopper.json.Statement.Op.MULTIPLY;

import com.graphhopper.GraphHopper;
import com.graphhopper.config.Profile;
import com.graphhopper.matching.EdgeMatch;
import com.graphhopper.matching.MapMatching;
import com.graphhopper.matching.MatchResult;
import com.graphhopper.matching.Observation;
import com.graphhopper.util.CustomModel;
import com.graphhopper.util.EdgeIteratorState;
import com.graphhopper.util.PMap;
import com.graphhopper.util.shapes.GHPoint;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class GraphHopperRunner {

    public static void main(String[] args) throws Exception {
        if (args.length < 8) {
            System.err.println("usage: GraphHopperRunner <pbf> <gh_dir> <traces_tsv> <out_tsv>"
                    + " <vehicle> <sigma_m> <min_network_size> <workers>");
            System.exit(2);
        }
        String pbf = args[0];
        String ghDir = args[1];
        String tracesPath = args[2];
        String outPath = args[3];
        String vehicle = args[4];
        double sigma = Double.parseDouble(args[5]);
        int minNetwork = Integer.parseInt(args[6]);
        int workers = Math.max(1, Integer.parseInt(args[7]));

        // ---- Build (or load cached) routing graph -----------------------------
        GraphHopper hopper = new GraphHopper();
        hopper.setOSMFile(pbf);
        hopper.setGraphHopperLocation(ghDir);
        hopper.setEncodedValuesString(vehicle + "_access, " + vehicle + "_average_speed");
        // Access + speed custom model for the chosen vehicle: block edges the vehicle
        // cannot use (priority 0) and cap speed at its per-edge average. (This is the
        // same model as the core TestProfiles.accessAndSpeed helper, inlined so the
        // runner depends only on stable public API, not a test-named utility class.)
        CustomModel customModel = new CustomModel()
                .addToPriority(If("!" + vehicle + "_access", MULTIPLY, "0"))
                .addToSpeed(If("true", LIMIT, vehicle + "_average_speed"));
        hopper.setProfiles(new Profile("profile").setCustomModel(customModel));
        // Keep every edge snappable: no subnetwork pruning (mirrors Valhalla, which
        // snaps to anything) so recall is not capped by dropped small components.
        hopper.setMinNetworkSize(minNetwork);
        // Import all highway classes (default is already empty; set explicitly so a
        // GraphHopper default change can't silently drop footways/sidewalks).
        hopper.getReaderConfig().setIgnoredHighways(new ArrayList<>());
        hopper.importOrLoad();

        // ---- Read all traces --------------------------------------------------
        List<String[]> traces = new ArrayList<>();  // [id, coordsCsv]
        try (BufferedReader br = Files.newBufferedReader(Paths.get(tracesPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                int tab = line.indexOf('\t');
                if (tab < 0) continue;
                traces.add(new String[]{line.substring(0, tab), line.substring(tab + 1)});
            }
        }

        final GraphHopper gh = hopper;
        // MapMatching is not safe to share across threads; each worker gets its own
        // instance built from the same read-only graph (like Meili's per-thread Actor).
        ThreadLocal<MapMatching> localMM = ThreadLocal.withInitial(() -> {
            PMap hints = new PMap().putObject("profile", "profile");
            MapMatching mm = MapMatching.fromGraphHopper(gh, hints);
            mm.setMeasurementErrorSigma(sigma);
            return mm;
        });

        ConcurrentLinkedQueue<String> out = new ConcurrentLinkedQueue<>();
        AtomicInteger matchedTargets = new AtomicInteger(0);
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        for (String[] t : traces) {
            pool.submit(() -> {
                String id = t[0];
                List<Observation> obs = parseTrace(t[1]);
                if (obs.size() < 2) return;
                Map<String, double[]> perWay = new HashMap<>();  // way_id -> [sum_m, sum_states]
                try {
                    MatchResult mr = localMM.get().match(obs);
                    for (EdgeMatch em : mr.getEdgeMatches()) {
                        EdgeIteratorState e = em.getEdgeState();
                        String name = e.getName();  // synthetic way_id (see convert/pbf.py)
                        if (name == null || name.isEmpty()) continue;
                        double[] agg = perWay.computeIfAbsent(name, k -> new double[2]);
                        agg[0] += e.getDistance();
                        agg[1] += em.getStates().size();
                    }
                } catch (Exception ex) {
                    // GraphHopper throws when a trace cannot be snapped (an unmatched
                    // local segment) — a legitimate no-match, not a runner error.
                    return;
                }
                if (perWay.isEmpty()) return;
                StringBuilder sb = new StringBuilder(id).append('\t');
                boolean first = true;
                for (Map.Entry<String, double[]> en : perWay.entrySet()) {
                    if (!first) sb.append(';');
                    sb.append(en.getKey()).append(',').append(en.getValue()[0])
                      .append(',').append((int) en.getValue()[1]);
                    first = false;
                }
                out.add(sb.toString());
                matchedTargets.incrementAndGet();
            });
        }
        pool.shutdown();
        pool.awaitTermination(6, TimeUnit.HOURS);

        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(outPath))) {
            for (String line : out) {
                bw.write(line);
                bw.newLine();
            }
        }
        System.err.println("GraphHopperRunner: matched " + matchedTargets.get()
                + " / " + traces.size() + " target traces");
        hopper.close();
    }

    private static List<Observation> parseTrace(String coordsCsv) {
        List<Observation> obs = new ArrayList<>();
        for (String p : coordsCsv.split(";")) {
            int c = p.indexOf(',');
            if (c < 0) continue;
            double lon = Double.parseDouble(p.substring(0, c));
            double lat = Double.parseDouble(p.substring(c + 1));
            obs.add(new Observation(new GHPoint(lat, lon)));
        }
        return obs;
    }
}
