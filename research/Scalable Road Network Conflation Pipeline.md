# **Scalable Vector-to-Vector Road Network Conflation: A Comprehensive Architectural Analysis and Research Review**

## **1\. Introduction: The Conflation Imperative in the Age of Open Data**

The geospatial domain is currently navigating a profound structural shift characterized by the simultaneous proliferation of high-frequency crowd-sourced data and the rigorous standardization of global mapping frameworks. Historically, the maintenance of road network databases was the exclusive province of centralized authorities—municipal governments, national mapping agencies, and proprietary vendors. These authoritative datasets are characterized by high attribute precision and legal validation but often suffer from significant temporal latency and spatial fragmentation. Conversely, crowd-sourced platforms, most notably OpenStreetMap (OSM), have demonstrated an unprecedented capacity for rapid update cycles and the mapping of micromobility infrastructure, yet they frequently lack the consistent schema enforcement required for enterprise-grade applications.

"Conflation" is the technical discipline addressing this dichotomy. Defined as the automated process of combining geographic information from overlapping sources to retain accurate data, minimize redundancy, and reconcile conflicts, conflation is far more complex than simple map layering.1 In the specific context of road networks, vector-to-vector conflation requires the algorithmic fusion of two distinct graph structures—a Reference graph (often the authoritative or standard layer) and a Target graph (the source of new or richer data)—to produce a unified derivative dataset.

The emergence of the Overture Maps Foundation has fundamentally altered the strategic objectives of conflation. By establishing a common schema and the Global Entity Reference System (GERS), Overture has transitioned the industry's focus from merely merging geometries to "linking" disparate datasets to a persistent global backbone.3 This shift aims to eliminate the "Conflation Tax"—the massive resource drain organizations incur by repeatedly matching data—by establishing stable identifiers that persist across data releases.5

This report provides an exhaustive analysis of the methodologies, algorithms, and architectural patterns necessary to build a scalable, open-source road network conflation pipeline. Unlike traditional desktop-based workflows or monolithic C++ applications like Hootenanny, the proposed architecture leverages the distributed computing power of Apache Spark (PySpark) and the orchestration capabilities of Apache Airflow. It rigorously addresses the challenges of "spaghetti" inputs (non-topological geometry), avoids reliance on GPS traces in favor of static graph alignment, and integrates advanced Machine Learning (ML) paradigms including Graph Neural Networks (GNNs) to achieve high-fidelity matching.6

## **2\. Theoretical Framework of Vector Conflation**

To engineer a robust pipeline, one must first deconstruct the theoretical underpinnings of road network matching. This involves concepts from computational geometry, graph theory, and semantic ontology.

### **2.1 The Graph Isomorphism Problem**

Fundamentally, road network matching is a variation of the Subgraph Isomorphism problem, which is known to be NP-hard. A road network is modeled as a graph $G \= (V, E)$, where $V$ represents vertices (intersections, dead-ends) and $E$ represents edges (road segments).8 Conflation seeks a mapping function $f: G\_1 \\rightarrow G\_2$ such that the structural and attribute relationships in the source graph $G\_1$ are preserved in the target graph $G\_2$.

However, unlike the exact isomorphism required in chemical substructure searching, geospatial graph matching is probabilistic and error-tolerant. Two graphs representing the same physical road network will rarely be isomorphic due to:

* **Geometric Distortion:** Different survey methods, projection errors, or tectonic shifts can cause the "same" intersection to appear at coordinates shifted by meters or decimeters.6  
* **Topological Abstraction:** One dataset might model a dual-carriageway boulevard as two parallel one-way edges, while another models it as a single bi-directional centerline. A roundabout might be represented as a loop of edges or a single point node.9  
* **Granularity Mismatch:** A "long" edge in a simplified highway network might correspond to a chain of ten shorter edges in a detailed local municipal dataset.10

Therefore, the problem is reformulated as **Maximum Weight Matching** or **Graph Alignment**, where the goal is to maximize a global similarity score based on feature vectors associated with nodes and edges.11

### **2.2 The "Spaghetti" Data Challenge**

A critical constraint identified for this pipeline is the handling of non-topological inputs. Many authoritative datasets (e.g., legacy Shapefiles from local governments) are stored as "spaghetti" data. In this format, geometric LineStrings may visually cross on a map, but there is no explicit node at the intersection point. Topological connectivity is implicit rather than explicit.9

This presents a severe obstacle for graph-based matching algorithms, which rely on traversing edges from node to node. If a road segment physically intersects another but does not share a vertex ID, graph traversal algorithms (like BFS or Dijkstra) will fail to see the connection. Consequently, a "Topology Estimation" or "Planarization" phase is a strict prerequisite for the pipeline. This process involves detecting all geometric intersections, splitting the LineStrings at these points, and generating new nodes to create a planar graph where edges intersect only at vertices.13

### **2.3 The Tripartite Feature Space**

Successful automated matching relies on quantifying similarity across three orthogonal dimensions. Relying on any single dimension is insufficient for high-precision conflation.

#### **2.3.1 Geometric Similarity**

Geometric measures quantify the spatial proximity and shape alignment of features in Euclidean space.

* **Hausdorff Distance:** This metric measures the "worst-case" mismatch between two shapes. For two road segments, it represents the maximum distance one would have to travel from a point on one segment to the nearest point on the other. It is particularly useful for bounding the error of a match.7  
* **Fréchet Distance:** Often described as the "dog-walking distance," this metric accounts for the sequence and direction of points along the curves. Unlike Hausdorff distance, which treats lines as point sets, Fréchet distance respects the flow of the geometry, making it superior for distinguishing between parallel roads (e.g., a frontage road vs. a highway).15  
* **Buffer Intersection over Union (IoU):** A raster-proxy method where lines are buffered (e.g., by 10 meters) to create polygons. The ratio of the intersection area to the union area provides a robust scalar measure of overlap.16  
* **Heading/Azimuth Delta:** The angular difference between road segments is critical for resolving intersections. A road matching candidate with high spatial proximity but a 90-degree heading difference is likely a crossing street, not a match.9

#### **2.3.2 Topological Similarity**

Topology describes the connectivity structure of the network, which remains invariant under rubber-sheeting or elastic deformation.

* **Node Valence (Degree):** The count of edges incident to a node. A 4-way intersection in the Reference graph is statistically unlikely to match a 2-way pseudo-node in the Target graph. Matching based on valence distribution is a powerful heuristic for anchoring the graph.16  
* **Centrality Measures:** Metrics such as Betweenness Centrality or PageRank identify the "structural importance" of a road. Main arterials will exhibit high centrality scores in both datasets, providing a strong signal for matching even if attribute names differ.17  
* **Graphlet Signatures:** Small, induced subgraphs (graphlets) surrounding a node act as a "topological fingerprint." For instance, a node involved in a triangle cycle (common in highway ramps) has a distinct signature compared to a node in a gridiron lattice.6

#### **2.3.3 Semantic Similarity**

Semantic attributes serve as the final disambiguation layer.

* **String Distance:** Algorithms like Levenshtein, Jaro-Winkler, or N-Gram cosine similarity quantify the resemblance of street names (e.g., "Main St" vs. "N Main Street").  
* **Ontology Alignment:** Normalizing disparate classification schemas (e.g., OSM highway=primary vs. TIGER MTFCC=S1100) to a common ontology—such as the Overture Maps schema—enables categorical comparison.18

## **3\. Analysis of Open Source Methodologies**

To design a superior pipeline, we must analyze existing open-source tools to understand their strengths (logic) and weaknesses (architecture).

### **3.1 Hootenanny: The Algorithmic Gold Standard**

**Hootenanny** stands as the most mature open-source conflation platform, originally developed for the National Geospatial-Intelligence Agency (NGA). It encapsulates decades of research into road matching logic.2

#### **3.1.1 The "Unifying Roads" Algorithm**

This algorithm represents a shift from heuristic rules to machine learning. It functions by generating a complex feature vector for every candidate pair, which includes:

* **Weighted Shape Distance:** A metric derived from Savary (2005) that balances coordinate proximity with shape fidelity.7  
* **Circular Error:** Using the reported accuracy of the source data to define dynamic search radii.7  
* **Topology Score:** Assessing the compatibility of the connectivity graph at the endpoints of the candidate lines.

The algorithm uses a pre-trained model (typically a Random Forest variant) to classify pairs into "Match," "Miss," or "Review." The "Review" class triggers an interactive conflict resolution workflow, highlighting the tool's focus on human-in-the-loop cartography.7

#### **3.1.2 The "Network Roads" Algorithm**

This alternative approach utilizes graph traversal. It identifies "seed" matches—pairs with extremely high confidence (e.g., unique name \+ exact topology)—and then "grows" the solution outward along the graph edges. This "Subgraph Growing" technique is highly effective for maintaining the continuity of long linear features like highways.7

#### **3.1.3 Architectural Limitations for Spark**

While algorithmically sound, Hootenanny is a monolithic C++ application with a heavy Qt dependency. Integrating it into a PySpark pipeline typically involves "shelling out" to the binary via pipe(), which incurs serialization overhead and treats the matcher as a black box. A scalable native Spark pipeline should **port the logic** of Hootenanny's feature engineering into PySpark UDFs rather than wrapping the executable.

### **3.2 MAYUR: Optimization-Based Matching**

Research from the University of British Columbia introduces **MAYUR**, which frames map matching as a "Rank Join" problem in databases.6 Instead of classifying pairs in isolation, it seeks to find a set of matches that maximizes a global scoring function, respecting the connectivity of the road network.

* **Rank Join Adaptation:** MAYUR treats each edge of the reference graph as a relation and uses a rank-aware join algorithm to find the best matching subgraph in the target database.  
* **Scalability:** By introducing early pruning optimizations, MAYUR claims to outperform Hootenanny in efficiency for large relation sets. This "global optimization" philosophy is crucial for resolving the 1-to-N matching ambiguities common in road networks.6

### **3.3 Apache Sedona (GeoSpark): The Geometric Engine**

**Apache Sedona** (formerly GeoSpark) is the foundational technology for the proposed pipeline. It extends the Apache Spark core to support spatial data types and operations.23

* **Spatial RDDs & Partitioning:** Sedona addresses the "data skew" problem in distributed spatial joins. By employing partitioning schemes like Quad-Tree or KDB-Tree, it clusters spatially proximate features onto the same physical worker nodes. This minimizes the network shuffle required for operations like "find all target roads within 50 meters of this reference road".25  
* **Topology Primitives:** Sedona provides the low-level geometric functions required for topology estimation, such as ST\_Intersection (to find split points), ST\_MakeValid (to fix bow-ties and self-intersections), and ST\_Union (to node spaghetti lines).14

### **3.4 GraphFrames: The Topological Engine**

**GraphFrames** provides the graph analytic capabilities lacking in pure geometry engines. By converting Sedona DataFrames into a Vertex/Edge graph structure, the pipeline can execute complex graph algorithms natively on Spark.17

* **Motif Finding:** This DSL allows users to query for specific structural patterns, such as (a)-\[e\]-\>(b); (b)-\[e2\]-\>(c);\!(a)--\>(c), which effectively searches for specific road configurations.  
* **Connected Components:** Essential for quality assurance, detecting if the conflated network has become fragmented into disconnected islands.17

## **4\. Advanced Matching Paradigms: Machine Learning and GNNs**

To satisfy the requirement for utilizing ML models that ingest topology, classification, and geometry, the pipeline must move beyond simple regression or decision trees toward structure-aware learning.

### **4.1 Graph Neural Networks (GNNs) for Alignment**

Graph Neural Networks represent the cutting edge of network alignment research. Unlike traditional ML that treats rows as independent instances, GNNs learn from the *structure* of the data.

* **Siamese GNN Architecture:** A proven approach for graph alignment is the Siamese GNN.29 In this architecture, two identical GNNs (sharing weights) process the Reference Graph and the Target Graph independently. The GNN layers (e.g., GraphSAGE or GAT) aggregate information from a node's neighbors to generate a dense vector embedding for each node.  
* **Contrastive Learning:** The model is trained using a contrastive loss function (e.g., Triplet Loss). The objective is to minimize the Euclidean distance between the embeddings of "true match" node pairs ($u \\in G\_{Ref}, v \\in G\_{Tgt}$) while pushing apart the embeddings of non-matching pairs.30  
* **Cross-Graph Attention:** More sophisticated models, such as the "Contextual Alignment Enhanced Cross Graph Attention Network," incorporate an attention mechanism that allows the embedding process of one graph to be influenced by the features of the other. This helps the model align heterogeneous structures, such as matching a complex multi-ramp interchange in the Reference graph to a simplified intersection node in the Target graph.31

### **4.2 Sequence-to-Sequence Models (DeepMapMatch)**

While typically applied to trajectory matching, Sequence-to-Sequence (Seq2Seq) models offer a novel way to handle the "sequence" aspect of roads. **DeepMapMatch** uses Recurrent Neural Networks (RNNs) to map a sequence of noisy GPS points to a sequence of road segments.33

* **Adaptation for Vector Conflation:** By treating a linear road feature (e.g., "Main Street") as a sequence of segments and intersections, a Seq2Seq model (or a Transformer) can be trained to "translate" a path from the Target graph domain to the Reference graph domain. This is particularly valuable for handling "spaghetti" lines that may cover multiple topological edges in the reference graph.

### **4.3 Feature Engineering Strategy**

Based on the analysis of Hootenanny 7 and academic literature 16, the ML model in the pipeline should be fed a comprehensive feature vector.

**Table 1: Proposed Feature Set for ML Matching Model**

| Feature Domain | Feature Name | Description & Relevance | Computation Engine |
| :---- | :---- | :---- | :---- |
| **Geometry** | **Hausdorff Distance** | Max deviation between line geometries. Bounds the worst-case error. | Sedona ST\_HausdorffDistance |
|  | **Fréchet Distance** | "Dog-walking" distance. Captures shape flow and ordering better than Hausdorff. | Python UDF (libs like similaritymeasures) |
|  | **Buffer IoU** | Intersection-over-Union of buffered polygons. Robust to minor shifts. | Sedona ST\_Buffer, ST\_Area |
|  | **Angle Delta** | Difference in overall heading/azimuth. Distinguishes parallel vs. crossing roads. | Spark SQL Math |
|  | **Projection Distance** | Average perpendicular distance from vertices of Line A to Line B. | Sedona Linear Referencing |
| **Topology** | **Degree Divergence** | Difference in node valence (degree) at endpoints. | GraphFrames degrees |
|  | **PageRank Delta** | Difference in centrality scores. Prevents matching major roads to minor ones. | GraphFrames pageRank |
|  | **Graphlet Vector** | Count of local substructures (triangles, stars) at the node. | GraphFrames Motif Finding |
| **Semantics** | **Name Similarity** | Levenshtein, Jaro-Winkler, or Soundex score of street names. | Spark MLlib / fuzzywuzzy |
|  | **Class Alignment** | One-hot encoded match of road hierarchy (e.g., Primary vs. Secondary). | Spark ML Vectors |
|  | **Embedding Cosine** | Similarity of Node2Vec or GNN-learned embeddings. | Spark ML / PyTorch |

## **5\. Architectural Design: The Scalable Pipeline**

This section details a cloud-native architecture using **PySpark** on a cluster (e.g., Databricks, EMR) orchestrated by **Apache Airflow**. The design philosophy emphasizes modularity, allowing the "Matching Engine" to be swapped (e.g., from XGBoost to GNN) without refactoring the data ingestion or export logic.

### **5.1 Architecture Overview**

The pipeline is structured as a Directed Acyclic Graph (DAG) in Airflow with six distinct stages.

1. **Ingest & Normalization:** Loading disparate formats and mapping to the Overture Schema.  
2. **Topology Reconstruction:** The "Planarization" of spaghetti data.  
3. **Graph Construction & Feature Engineering:** Calculating embeddings and metrics.  
4. **Blocking (Candidate Generation):** Efficient spatial indexing to reduce search space.  
5. **Matching Model (The Brain):** ML inference.  
6. **Resolution & GERS Linkage:** Graph consistency checks and ID assignment.

### **5.2 Stage 1: Ingest & Normalization (PySpark \+ Sedona)**

* **Reference Input:** The Overture Maps Transportation theme (GeoParquet). This data is already topologically clean and contains connector (nodes) and segment (edges).36  
* **Target Input:** Local authoritative data (e.g., City of New York Street Centerlines) in Shapefile or GeoJSON format.  
* **Process:**  
  * Use Sedona's ShapefileReader or GeoParquetReader to load data into a Spatial DataFrame.  
  * **Schema Mapping:** Use PySpark transformations to map local attributes (e.g., L\_STNAME, SPEED) to Overture's schema (roadName, speedLimits). This standardization is vital for the semantic feature extraction phase.  
  * **Projection:** Reproject both datasets to a common coordinate reference system (CRS). While Overture uses WGS84 (EPSG:4326), metric calculations (length, buffer) require a projected CRS (e.g., UTM) or the use of Sedona's spheroid-aware functions like ST\_DistanceSpheroid.37

### **5.3 Stage 2: Topology Estimation (The "Spaghetti" Fix)**

This stage addresses the user's requirement to handle non-topological inputs. The goal is to convert visual intersections into logical nodes.

**Algorithm 1: Distributed Planarization in PySpark/Sedona**

1. **Explode Multi-Geometries:** Use ST\_Dump logic to decompose any MultiLineString features into simple LineStrings.38  
2. **Intersection Detection:** Perform a spatial self-join on the Target DataFrame using ST\_Intersects (or ST\_Crosses) to identify all points where lines physically cross.  
   Python  
   intersections \= lines\_df.alias("a").join(lines\_df.alias("b"),  
       expr("ST\_Crosses(a.geometry, b.geometry)")) \\  
      .select(expr("ST\_Intersection(a.geometry, b.geometry) as split\_point"))

3. **Node Accumulation:** Create a unified DataFrame of Nodes by combining the original start/end points of all lines with the newly discovered split\_points.  
4. **Line Splitting:** This is the most computationally intensive step. The original lines must be split at the locations of the Nodes. In a distributed environment, this is best handled by partitioning the lines spatially (Quad-Tree) and running a local splitting algorithm (using JTS/Shapely) inside a mapPartitions function to avoid global shuffling.  
5. **Snapping:** Apply a spatial window function or ST\_Snap with a small tolerance (e.g., 0.5 meters) to merge nodes that are artificially separated due to precision errors (undershoots/overshoots).9

### **5.4 Stage 3: Feature Engineering & Graph Construction**

Once the topology is established, the pipeline extracts the features required for the ML model.

* **GraphFrames Integration:** Convert the Node and Edge DataFrames into a GraphFrame.  
  * Run ConnectedComponents to identify disconnected subgraphs (islands). This serves as a quality check; a healthy road network typically has one massive giant component.17  
  * Run PageRank to compute node centrality.  
* **Embedding Generation (Node2Vec):**  
  * Implement **Node2Vec** on Spark (available via Spark-MLlib extensions). This algorithm performs random walks on the graph to generate a sequence of nodes, which are then fed into a Word2Vec model.  
  * **Result:** A dense vector for each node that encodes its structural role (e.g., "hub," "bridge," "dead-end"). This allows the ML model to match nodes based on "structural equivalence" rather than just coordinates.35

### **5.5 Stage 4: Candidate Generation (Blocking)**

Comparing every edge in the Reference graph to every edge in the Target graph is an $O(N \\times M)$ operation, which is intractable for large datasets. A "Blocking" strategy is required to reduce the search space.

* **S2/H3 Cell Indexing:** Assign every edge to a set of covering S2 cells (e.g., Level 13 or 14).  
* **Explode & Join:** Explode the edges by their cell IDs and perform a join on Cell\_ID. This effectively buckets edges into small geospatial partitions.  
* **Buffer Growing Alternative:** Use Sedona's ST\_Buffer on the Reference graph edges and perform a range join to find all Target edges that intersect the buffer.  
* **Coarse Filtering:** Apply inexpensive filters immediately after the join to discard obvious non-matches (e.g., heading difference \> 45 degrees, length ratio \> 5.0).

### **5.6 Stage 5: The Matching Model**

This stage applies the predictive model to the candidate pairs.

* **Tier 1: Gradient Boosted Trees (XGBoost/LightGBM on Spark):**  
  * This is the recommended starting point. XGBoost scales natively on Spark and handles tabular features (distances, scores) exceptionally well.  
  * It is interpretable (feature importance plots) and robust to missing attributes.  
* **Tier 2: Graph Neural Networks (GNN):**  
  * For complex scenarios where geometry is unreliable, a GNN approach is superior.  
  * **Implementation:** Use a library like **BigDL** or **Spark-Torch** to run distributed inference. The GNN takes the graph structure (adjacency matrix) and node features to predict the likelihood of a link between a Reference Node and a Target Node.

### **5.7 Stage 6: Conflict Resolution & GERS Assignment**

The final stage resolves ambiguities and generates the output.

* **1:N Resolution:** Road matching is often one-to-many (e.g., one Overture segment matches five short local segments). The logic must group these matches. If Target Edge $T$ matches Reference Edges $R\_1, R\_2$, the system checks if $R\_1$ and $R\_2$ are connected in the Reference Graph. If they are, the match is valid.  
* **GERS Linkage:**  
  * **Matches:** The Overture gers\_id is assigned to the local feature.  
  * **No-Matches:** Local features with no confident match are flagged. If they pass validity checks (e.g., connected to the main graph, significant length), they represent "missing data" in Overture and are candidates for contribution (new GERS ID generation).  
* **Bridge File Generation:** The output is a "Bridge File" (Parquet format) containing columns: {local\_id, gers\_id, match\_confidence, match\_type}. This file allows the local dataset to be joined with any other GERS-enabled dataset in the future.4

## **6\. Operationalizing Overture Maps and GERS**

### **6.1 The GERS Philosophy: Linking vs. Merging**

The Overture Maps Foundation advocates for a paradigm shift from "Merging" to "Linking." In a traditional merge, attributes from both sources are blended into a single new geometry, destroying the provenance of the original data. In the GERS paradigm, the Overture geometry serves as the stable spatial backbone. Local data is "hung" onto this backbone via the GERS ID.5

### **6.2 Managing GERS Stability**

GERS IDs are designed to be stable, but they can change (e.g., if a road is split or realigned). The pipeline must consume **Overture Data Changelogs**.

* **Process:** When a new Overture release drops, the pipeline should check the Changelog for any GERS IDs present in the Bridge File.  
* **Updates:** If a GERS ID has been split (one ID becomes two), the pipeline must re-evaluate the match for the associated local features. This ensures the Bridge File remains valid over time without re-running the expensive full conflation.4

## **7\. Case Study: Hypothetical Workflow**

Consider a Department of Transportation (DOT) providing a Shapefile of snowplow routes (high attribute detail, varying geometry) to be matched with Overture.

1. **Ingest:** The Shapefile is loaded into PySpark. It contains MultiLineStrings for entire routes.  
2. **Topology:** Sedona splits these routes at every intersection, converting a 10km "Route A" into 50 distinct edges.  
3. **Blocking:** S2 indexing groups these 50 edges with 200 nearby Overture segments.  
4. **Matching:** The XGBoost model analyzes pairs. It sees that "Local Edge 1" and "Overture Segment X" have a Hausdorff distance of 2m, parallel headings, and name similarity ("Main St" vs "Main Street"). It assigns a probability of 0.98.  
5. **Resolution:** The pipeline confirms that the sequence of local edges matches a contiguous sequence of Overture segments.  
6. **Output:** A Bridge File is produced. The DOT can now visualize their snowplow status on a global Overture basemap by joining on gers\_id, without needing to host their own map tiles.

## **8\. Conclusion and Future Directions**

The conflation of road networks at scale is no longer a problem of simple geometric overlay; it is a complex data engineering challenge that requires a synthesis of distributed computing, graph theory, and machine learning. By adopting an architecture built on **Apache Sedona** for geometry, **GraphFrames** for topology, and **Spark ML/GNNs** for probabilistic matching, organizations can process massive datasets that were previously intractable.

The integration of the **Overture Maps GERS** elevates this pipeline from a proprietary tool to a node in the global geospatial ecosystem. By producing standardized Bridge Files, this architecture minimizes the "Conflation Tax," allowing organizations to focus on deriving insights from their data rather than struggling to align it. Future work lies in the refinement of GNN architectures to handle increasingly complex urban topologies and the development of real-time conflation capabilities for streaming sensor data.

## **9\. Recommendation Summary**

* **Core Engine:** Apache Spark (PySpark) \+ Apache Sedona.  
* **Orchestration:** Apache Airflow with dynamic task mapping.  
* **Matching Logic:** Port Hootenanny's feature engineering (Hausdorff, Weighted Shape Distance) to PySpark UDFs.  
* **Model:** Start with XGBoost; evolve to Siamese GNNs for complex cases.  
* **Target Output:** GERS-keyed Bridge Files (Parquet).  
* **Topology Handling:** Explicit Planarization step using Sedona ST\_Intersection and ST\_Split.

#### **Works cited**

1. Conflation \- OpenStreetMap Wiki, accessed January 10, 2026, [https://wiki.openstreetmap.org/wiki/Conflation](https://wiki.openstreetmap.org/wiki/Conflation)  
2. Conflation, accessed January 10, 2026, [http://52north.github.io/wps-profileregistry/concept/conflation.html](http://52north.github.io/wps-profileregistry/concept/conflation.html)  
3. Overture Maps Documentation: Introduction, accessed January 10, 2026, [https://docs.overturemaps.org/](https://docs.overturemaps.org/)  
4. GERS Tutorial | Overture Maps Documentation, accessed January 10, 2026, [https://docs.overturemaps.org/gers/gers-tutorial/](https://docs.overturemaps.org/gers/gers-tutorial/)  
5. Understanding Overture's Global Entity Reference System, accessed January 10, 2026, [https://overturemaps.org/blog/2025/understanding-overtures-global-entity-reference-system/](https://overturemaps.org/blog/2025/understanding-overtures-global-entity-reference-system/)  
6. A principled approach to automated road network conflation \- UBC Library Open Collections, accessed January 10, 2026, [https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0398182](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0398182)  
7. hootenanny/docs/algorithms/RoadConflation.asciidoc at master \- GitHub, accessed January 10, 2026, [https://github.com/ngageoint/hootenanny/blob/master/docs/algorithms/RoadConflation.asciidoc](https://github.com/ngageoint/hootenanny/blob/master/docs/algorithms/RoadConflation.asciidoc)  
8. Line Structure Representation for Road Network Analysis \- Journal of Transport and Land Use, accessed January 10, 2026, [https://www.jtlu.org/index.php/jtlu/article/view/744/1364](https://www.jtlu.org/index.php/jtlu/article/view/744/1364)  
9. Methods and Implementations of Road-Network Matching \- mediaTUM, accessed January 10, 2026, [https://mediatum.ub.tum.de/doc/820125/820125.pdf](https://mediatum.ub.tum.de/doc/820125/820125.pdf)  
10. An Area Partitioning and Subgraph Growing (APSG) Approach to the Conflation of Road Networks \- PMC \- NIH, accessed January 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8877969/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8877969/)  
11. Dynamic Knowledge Graph Alignment, accessed January 10, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/16585/16392](https://ojs.aaai.org/index.php/AAAI/article/view/16585/16392)  
12. Topological Graph Simplification Solutions to the Street Intersection Miscount Problem \- arXiv, accessed January 10, 2026, [https://arxiv.org/pdf/2407.00258](https://arxiv.org/pdf/2407.00258)  
13. convert a shapefile into graph with nodes and edges \- GIS StackExchange, accessed January 10, 2026, [https://gis.stackexchange.com/questions/99609/convert-a-shapefile-into-graph-with-nodes-and-edges](https://gis.stackexchange.com/questions/99609/convert-a-shapefile-into-graph-with-nodes-and-edges)  
14. Apache Sedona Spatial Joins with Spark, accessed January 10, 2026, [https://sedona.apache.org/latest/tutorial/concepts/spatial-joins/](https://sedona.apache.org/latest/tutorial/concepts/spatial-joins/)  
15. Compute distance with Sedona and Apache Spark, accessed January 10, 2026, [https://sedona.apache.org/latest/tutorial/concepts/distance-spark/](https://sedona.apache.org/latest/tutorial/concepts/distance-spark/)  
16. Conflation of Road Networks from Digital Maps \- mediaTUM, accessed January 10, 2026, [https://mediatum.ub.tum.de/doc/1310567/1310567.pdf](https://mediatum.ub.tum.de/doc/1310567/1310567.pdf)  
17. Graph-Based Data and GraphFrames in PySpark — Day 32 of 100 Days of Data Engineering, AI and Azure Challenge | by Karthik | Medium, accessed January 10, 2026, [https://medium.com/@krthiak/graph-based-data-and-graphframes-in-pyspark-day-32-of-100-days-of-data-engineering-ai-and-azure-449da4a628e7](https://medium.com/@krthiak/graph-based-data-and-graphframes-in-pyspark-day-32-of-100-days-of-data-engineering-ai-and-azure-449da4a628e7)  
18. Transportation schema concepts | Overture Maps Documentation, accessed January 10, 2026, [https://docs.overturemaps.org/schema/concepts/by-theme/transportation/](https://docs.overturemaps.org/schema/concepts/by-theme/transportation/)  
19. Transportation theme navigates to GA | Overture Maps Documentation, accessed January 10, 2026, [https://docs.overturemaps.org/blog/2024/12/18/transportation-to-ga/](https://docs.overturemaps.org/blog/2024/12/18/transportation-to-ga/)  
20. Is there a conflation tool that can generate readable list of matches for review and sharing?, accessed January 10, 2026, [https://community.openstreetmap.org/t/is-there-a-conflation-tool-that-can-generate-readable-list-of-matches-for-review-and-sharing/104064](https://community.openstreetmap.org/t/is-there-a-conflation-tool-that-can-generate-readable-list-of-matches-for-review-and-sharing/104064)  
21. Hootenanny conflates multiple maps into a single seamless map. \- GitHub, accessed January 10, 2026, [https://github.com/ngageoint/hootenanny](https://github.com/ngageoint/hootenanny)  
22. KRAFT: A Knowledge Graph-Based Framework for Automated Map Conflation \- arXiv, accessed January 10, 2026, [https://arxiv.org/html/2509.04684v1](https://arxiv.org/html/2509.04684v1)  
23. Working with Apache Sedona \- Delta Lake, accessed January 10, 2026, [https://delta.io/blog/apache-sedona/](https://delta.io/blog/apache-sedona/)  
24. apache/sedona: A cluster computing framework for processing large-scale geospatial data \- GitHub, accessed January 10, 2026, [https://github.com/apache/sedona](https://github.com/apache/sedona)  
25. n0mer/GeoSpark: A Cluster Computing System for Processing Large-Scale Spatial Data \- GitHub, accessed January 10, 2026, [https://github.com/n0mer/GeoSpark](https://github.com/n0mer/GeoSpark)  
26. Distributed Graph Layout with Spark | Request PDF \- ResearchGate, accessed January 10, 2026, [https://www.researchgate.net/publication/281348264\_Distributed\_Graph\_Layout\_with\_Spark](https://www.researchgate.net/publication/281348264_Distributed_Graph_Layout_with_Spark)  
27. ST\_MakeValid \- PostGIS, accessed January 10, 2026, [https://postgis.net/docs/ST\_MakeValid.html](https://postgis.net/docs/ST_MakeValid.html)  
28. On-Time Flight Performance with GraphFrames for Apache Spark \- Databricks, accessed January 10, 2026, [https://www.databricks.com/blog/2016/03/16/on-time-flight-performance-with-graphframes-for-apache-spark.html](https://www.databricks.com/blog/2016/03/16/on-time-flight-performance-with-graphframes-for-apache-spark.html)  
29. SiG: A Siamese-Based Graph Convolutional Network to Align Knowledge in Autonomous Transportation Systems | Semantic Scholar, accessed January 10, 2026, [https://www.semanticscholar.org/paper/SiG%3A-A-Siamese-based-Graph-Convolutional-Network-to-Hao-Cai/67bb8140a8a5282061e84197cfa0b00bf0b600aa](https://www.semanticscholar.org/paper/SiG%3A-A-Siamese-based-Graph-Convolutional-Network-to-Hao-Cai/67bb8140a8a5282061e84197cfa0b00bf0b600aa)  
30. Using Graph Embedding Techniques in Process-Oriented Case-Based Reasoning \- MDPI, accessed January 10, 2026, [https://www.mdpi.com/1999-4893/15/2/27](https://www.mdpi.com/1999-4893/15/2/27)  
31. GTAT: empowering graph neural networks with cross attention \- PMC, accessed January 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11807142/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11807142/)  
32. A Contextual Alignment Enhanced Cross Graph Attention Network for Cross-lingual Entity Alignment \- ACL Anthology, accessed January 10, 2026, [https://aclanthology.org/2020.coling-main.520/](https://aclanthology.org/2020.coling-main.520/)  
33. vonfeng/DeepMapMatching: \[TMC 2020;SIGSPATIAL 2019\] DeepMM: Deep learning based map matching with data augmentation \- GitHub, accessed January 10, 2026, [https://github.com/vonfeng/DeepMapMatching](https://github.com/vonfeng/DeepMapMatching)  
34. NLP-enabled trajectory map-matching in urban road networks using transformer sequence-to-sequence model \- arXiv, accessed January 10, 2026, [https://arxiv.org/html/2404.12460v1](https://arxiv.org/html/2404.12460v1)  
35. Graph Embeddings for Street Network Analysis \- SNAP: Stanford, accessed January 10, 2026, [https://snap.stanford.edu/class/cs224w-2019/project/26424916.pdf](https://snap.stanford.edu/class/cs224w-2019/project/26424916.pdf)  
36. Transportation | Overture Maps Documentation, accessed January 10, 2026, [https://docs.overturemaps.org/guides/transportation/](https://docs.overturemaps.org/guides/transportation/)  
37. Predicate \- Apache Sedona, accessed January 10, 2026, [https://sedona.apache.org/latest/api/sql/Predicate/](https://sedona.apache.org/latest/api/sql/Predicate/)  
38. Work with GeoPandas and Shapely \- Apache Sedona, accessed January 10, 2026, [https://sedona.apache.org/latest/tutorial/geopandas-shapely/](https://sedona.apache.org/latest/tutorial/geopandas-shapely/)  
39. GERS \- Global Entity Reference System \- Overture Maps, accessed January 10, 2026, [https://overturemaps.org/gers/](https://overturemaps.org/gers/)