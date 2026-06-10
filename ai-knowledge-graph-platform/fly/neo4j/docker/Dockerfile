FROM neo4j:5.20-community
# Override listen addresses to bind on all interfaces including IPv6
# Required for Fly.io private network (6PN) which uses pure IPv6 routing
RUN echo 'server.bolt.listen_address=:::7687' >> /var/lib/neo4j/conf/neo4j.conf \
 && echo 'server.http.listen_address=:::7474' >> /var/lib/neo4j/conf/neo4j.conf
