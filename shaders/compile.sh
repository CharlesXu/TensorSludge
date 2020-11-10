#!/usr/bin/env bash
for x in *.comp; do
    glslc -O $x -o $x.spv &
done
wait
