# Build Hootenanny from source
# This takes 20-40 minutes on first build depending on cores

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
# Note: We need python2 because Hootenanny's configure script uses Python 2 syntax
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y \
    autoconf \
    autoconf-archive \
    automake \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libtool \
    libboost-all-dev \
    libcppunit-dev \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libgeos++-dev \
    libicu-dev \
    liblog4cxx-dev \
    libopencv-dev \
    libpq-dev \
    libproj-dev \
    libprotobuf-dev \
    libqt5sql5-psql \
    libstxxl-dev \
    nodejs \
    npm \
    protobuf-compiler \
    python2 \
    python3 \
    python3-pip \
    qtbase5-dev \
    qttools5-dev \
    ruby \
    ruby-dev \
    swig \
    unzip \
    libv8-dev \
    && ln -sf /usr/bin/python2 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Clone Hootenanny
WORKDIR /build
RUN git clone --depth 1 https://github.com/ngageoint/hootenanny.git

WORKDIR /build/hootenanny

# Run autotools to generate configure script, then build
RUN aclocal && autoconf && autoheader && automake --add-missing --copy || true
RUN ./configure --prefix=/opt/hootenanny --with-services=no || \
    (cat config.log && false)
RUN make -j$(nproc)
RUN make install

# Runtime image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libboost-filesystem1.74.0 \
    libboost-iostreams1.74.0 \
    libboost-program-options1.74.0 \
    libboost-system1.74.0 \
    libgdal30 \
    libgeos-c1v5 \
    libicu70 \
    liblog4cxx12 \
    libopencv-core4.5d \
    libpq5 \
    libproj22 \
    libprotobuf23 \
    libqt5core5a \
    libqt5network5 \
    libqt5sql5 \
    libqt5xml5 \
    libstxxl1v5 \
    && rm -rf /var/lib/apt/lists/*

# Copy built Hootenanny
COPY --from=builder /opt/hootenanny /opt/hootenanny

ENV PATH="/opt/hootenanny/bin:${PATH}"
ENV HOOT_HOME=/opt/hootenanny

WORKDIR /data
CMD ["hoot", "--version"]
