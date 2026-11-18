FROM postgres:16.6-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && python3 -m venv /opt/patroni-venv \
    && /opt/patroni-venv/bin/pip install --no-cache-dir "patroni[etcd3]==4.0.5" "psycopg[binary]==3.2.3" \
    && apt-get purge -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/patroni-venv/bin:$PATH"
USER postgres
ENTRYPOINT ["patroni"]
CMD ["/etc/patroni.yml"]
