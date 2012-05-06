adios_groupsize = 8 \
                + 8 \
                + 8 \
                + 1 * (chunk_size) \
                + 4 * (chunk_size) \
                + 4 * (chunk_size) \
                + 4 * (chunk_size) \
                + 4 * (chunk_size) \
                + 4 * (chunk_size) \
                + 4 * (chunk_size) \
                + 4 * (chunk_size) \
                + 1 * (chunk_size) \
                + 8 * (chunk_size) \
                + 8 * (chunk_size) \
                + 8 * (chunk_size) \
                + 8 \
                + 8 \
                + 8 \
                + 1 * (chunk_data_size);
adios_group_size (adios_handle, adios_groupsize, &adios_totalsize);
adios_write (adios_handle, "chunk_total", &chunk_total);
adios_write (adios_handle, "chunk_offset", &chunk_offset);
adios_write (adios_handle, "chunk_size", &chunk_size);
adios_write (adios_handle, "imageName", imageName);
adios_write (adios_handle, "tileSizeX", tileSizeX);
adios_write (adios_handle, "tileSizeY", tileSizeY);
adios_write (adios_handle, "tileOffsetX", tileOffsetX);
adios_write (adios_handle, "tileOffsetY", tileOffsetY);
adios_write (adios_handle, "channels", channels);
adios_write (adios_handle, "elemSize1", elemSize1);
adios_write (adios_handle, "cvDataType", cvDataType);
adios_write (adios_handle, "encoding", encoding);
adios_write (adios_handle, "id", id);
adios_write (adios_handle, "tile_offset", tile_offset);
adios_write (adios_handle, "tile_size", tile_size);
adios_write (adios_handle, "chunk_data_total", &chunk_data_total);
adios_write (adios_handle, "chunk_data_offset", &chunk_data_offset);
adios_write (adios_handle, "chunk_data_size", &chunk_data_size);
adios_write (adios_handle, "tiles", tiles);
