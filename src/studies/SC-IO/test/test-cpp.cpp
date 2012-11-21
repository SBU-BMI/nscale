/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */

#include <cstdio>
#include <cstring>
#include <stdint.h>


int main (int argc, char **argv) {

	 int32_t map[32];
	 printf("sizeof map = %ld, map[0] = %ld, value=%ld, pointer=%ld\n", sizeof(map), sizeof(map[0]), sizeof(*map), sizeof(&(*map)));
	    memset (map, -1, sizeof(map));
	
}
