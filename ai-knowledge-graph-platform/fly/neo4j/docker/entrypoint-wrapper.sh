#!/bin/bash
# Wrapper: run Neo4j entrypoint setup (config processing), then force IPv6 listen address
# before Neo4j actually starts.

# Let the original entrypoint generate neo4j.conf from env vars
# Run it in "configure only" mode by intercepting the exec at the end.

# Copy the original entrypoint and modify it to not exec neo4j at the end
CONF_FILE="/var/lib/neo4j/conf/neo4j.conf"

# Run original entrypoint just to process env vars into conf
/startup/docker-entrypoint.sh neo4j &
ENTRYPOINT_PID=$!

# Wait for conf file to be written (entrypoint writes it before starting neo4j)
for i in $(seq 1 30); do
  if grep -q "server.bolt.listen_address" "$CONF_FILE" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

# Kill the entrypoint before it starts neo4j
kill $ENTRYPOINT_PID 2>/dev/null
wait $ENTRYPOINT_PID 2>/dev/null

# Override bolt and http listen addresses with IPv6 wildcard
sed -i 's/^server\.bolt\.listen_address=.*/server.bolt.listen_address=[::]:7687/' "$CONF_FILE"
sed -i 's/^server\.http\.listen_address=.*/server.http.listen_address=[::]:7474/' "$CONF_FILE"

# Also ensure they exist if not present
grep -q "server.bolt.listen_address" "$CONF_FILE" || echo "server.bolt.listen_address=[::]:7687" >> "$CONF_FILE"
grep -q "server.http.listen_address" "$CONF_FILE" || echo "server.http.listen_address=[::]:7474" >> "$CONF_FILE"

# Now start neo4j directly
exec /var/lib/neo4j/bin/neo4j console
