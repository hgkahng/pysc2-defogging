#!/usr/bin/bash
echo "Download StarCraft II replays using the Blizzard Developer Portal API."

KEY="e7c4d3707de64a5483da2b6b5be0ebcd"
SECRET="mZhlhha5zRMpRGkQcLJIrA6L7VPZdnqR"
DOWNLOAD_DIRECTORY="X:/SC2Replays/download_cache/"
FILTER_VERSION="keep"

for CLIENT_VERSION in "4.0.2" "4.1.2" "4.6.1" "4.6.2" "4.7.1"; do
    REPLAYS_DIRECTORY="X:/SC2Replays/${CLIENT_VERSION}/"
    python scripts/download_replays.py \
        --key $KEY \
        --secret $SECRET \
        --version $CLIENT_VERSION \
        --replays_dir $REPLAYS_DIRECTORY \
        --download_dir $DOWNLOAD_DIRECTORY \
        --filter_version $FILTER_VERSION \
        # --extract
done
