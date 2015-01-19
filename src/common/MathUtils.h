/*
 * MathUtils.h
 *
 *  Created on: Jan 11, 2013
 *      Author: tcpan
 */

#ifndef MATHUTILS_H_
#define MATHUTILS_H_
#include <math.h>
#include <cstdlib>

namespace cci {
namespace common {

#ifdef _MSC_VER
	class	__declspec(dllexport) MathUtils {
#else
class MathUtils {
#endif
public:
	static double spare;
	static bool spareready;

	// http://en.wikipedia.org/wiki/Marsaglia_polar_method
	static double randn(const double &mean, const double &stdev) {
	    if (MathUtils::spareready) {
	    	MathUtils::spareready = false;
	            return MathUtils::spare * stdev + mean;
	    } else {
	            double u, v, s;
	            do {
	                    u = (double)rand()/(double)(RAND_MAX/2) - 1.0;
	                    v = (double)rand()/(double)(RAND_MAX/2) - 1.0;
	                    s = u * u + v * v;
	            } while (s >= 1 || s == 0);
	            double mul = sqrt(-2.0 * log(s) / s);
	            MathUtils::spare = v * mul;
	            MathUtils::spareready = true;
	            return mean + stdev * u * mul;
	    }
	};

};

} /* namespace common */
} /* namespace cci */
#endif /* MATHUTILS_H_ */
