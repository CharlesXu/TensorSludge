#!/usr/bin/env bash
for x in *.comp; do
    if [ $x != "in_size.comp" ]; then
        glslc -O $x -o $x.spv
    fi
done
