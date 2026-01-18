#!/usr/bin/env bash
set -euo pipefail

g++ -O2 -std=c++17 -pthread bench_client.cpp -o sedac_bench_client

