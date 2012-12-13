/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef _UTILS_LOGGER_H_
#define _UTILS_LOGGER_H_

#include <sys/time.h>
#include <tr1/unordered_map>
#include <vector>
#include <cmath>

#if defined (WITH_MPI)
#include "mpi.h"
#endif


namespace cci {
namespace common {

class event {
public:
	static const int COMPUTE;
	static const int MEM_IO;
	static const int GPU_MEM_IO;
	static const int NETWORK_IO;
	static const int NETWORK_WAIT;
	static const int FILE_I;
	static const int FILE_O;
	static const int ADIOS_INIT;
	static const int ADIOS_OPEN;
	static const int ADIOS_ALLOC;
	static const int ADIOS_WRITE;
	static const int ADIOS_CLOSE;
	static const int ADIOS_BENCH_OPEN;
	static const int ADIOS_BENCH_ALLOC;
	static const int ADIOS_BENCH_WRITE;
	static const int ADIOS_BENCH_CLOSE;
	static const int ADIOS_FINALIZE;
	static const int OTHER;

	event(const int _id, const std::string _name, const long long _start, const long long _end, const std::string _annotation,
			const int _type = -1) :
				id(_id), name(_name), starttime(_start), endtime(_end), annotation(_annotation), eventtype(_type) {};
	event(const int _id, const std::string _name, const long long _start, const std::string _annotation,
			const int _type = -1) :
				id(_id), name(_name), starttime(_start), endtime(-1), annotation(_annotation), eventtype(_type) {};
	virtual ~event() {};

	virtual std::string getAsString() const;

	virtual std::string getAsStringByType(const int &_type) const {
		if (eventtype == _type) {
			return getAsString();
		} else return std::string();
	};
	virtual std::string getAsStringByName(const std::string &_name) const {
		if (name.compare(_name) == 0) {
			return getAsString();
		} else return std::string();
	};
	virtual std::string getAsStringById(const int &_id) const {
		if (id == _id) {
			return getAsString();
		} else return std::string();
	};
	virtual std::string getName() const {
		return name;
	};
	virtual std::string getAnnotation() const {
		return annotation;
	};
	int getType() const {
		return eventtype;
	};
	int getId() const {
		return id;
	};
	long long getStart() const {
		return starttime;
	};
	long long getEnd() const {
		return endtime;
	};
	void setEnd(const long long &_end) {
		endtime = _end;
	};

	static inline long long timestampInUS()
	{
		struct timeval ts;
		gettimeofday(&ts, NULL);
		return (ts.tv_sec*1000000LL + (ts.tv_usec));
		 //   timespec ts;
		//    clock_gettime(CLOCK_REALTIME, &ts);
		//    return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
	};


private:
	int id;
	std::string name;
	long long starttime;
	long long endtime;
	int eventtype;
	std::string annotation;
};



class LogSession {

public :

	LogSession() : id(-1), name(std::string()), group(0), session_name(std::string()), start(0LL) {};
	LogSession(const int &_id, const std::string &_name, const int &_group,
			const std::string &_session_name, long long &_start);
	virtual ~LogSession();
	virtual void restart();
	virtual void log(cci::common::event e);

	virtual void toString(std::string &header, std::string &value);

	virtual void toOneLineString(std::string &value) ;

	virtual long getCountByEventName(std::string event_name) {
		return countByEventName[event_name];
	};
	virtual double getMeanByEventName(std::string event_name) {
		return double(sumDurationByEventName[event_name]) / double(countByEventName[event_name]);
	};
	virtual double getStdevByEventName(std::string event_name) {
		double mean = getMeanByEventName(event_name);
		long count =  countByEventName[event_name];
		if (count == 1) return -1;  //stdev for sample of 1 is undefined.
		return sqrt(double(sumSquareDurationByEventName[event_name]) / double(count) - mean * mean);
	};
	virtual long getCountByEventType(int event_type) {
		return countByEventType[event_type];
	};
	virtual double getMeanByEventType(int event_type) {
		return double(sumDurationByEventType[event_type]) / double(countByEventType[event_type]);
	};
	virtual double getStdevByEventType(int event_type) {
		double mean = getMeanByEventType(event_type);
		long count =  countByEventType[event_type];
		if (count == 1) return -1;  //stdev for sample of 1 is undefined.
		return sqrt(double(sumSquareDurationByEventType[event_type]) / double(count) - mean * mean);
	};

	virtual void toSummaryStringByName(std::string &header, std::string &value);
	virtual void toSummaryStringByType(std::string &header, std::string &value);

private :
	int id;
	std::string name;
	int group;
	std::string session_name;
	std::vector<cci::common::event> events;
	long long start;

	std::tr1::unordered_map<std::string, long> countByEventName;
	std::tr1::unordered_map<std::string, long long> sumDurationByEventName;
	std::tr1::unordered_map<std::string, long long> sumSquareDurationByEventName;

	std::tr1::unordered_map<int, long> countByEventType;
	std::tr1::unordered_map<int, long long> sumDurationByEventType;
	std::tr1::unordered_map<int, long long> sumSquareDurationByEventType;

};


/**
 *
 * holds on to the events in memory for multiple runs.  at the end, get all the data as a set of strings.
 *
 */
class Logger {
public :

	// _id is something like mpi rank or hostname
	Logger(const std::string &_logprefix, const int &_id, const std::string &_name, const int &_group) :
		id(_id), name(_name), group(_group), logprefix(_logprefix) {
		starttime = cci::common::event::timestampInUS();
		values.clear();
	};

	virtual ~Logger() {
		values.clear();
	};
	
	// session id is something like a filename or image name or hostname.
	virtual cci::common::LogSession* getSession(const std::string &session_name);

	virtual std::vector<std::string> toStrings();
	virtual std::vector<std::string> toOneLineStrings();
	virtual std::vector<std::string> toSummaryStringsByName();
	virtual std::vector<std::string> toSummaryStringsByType();
	virtual void write(const std::string &prefix);
	virtual void write();

#if defined (WITH_MPI)
	virtual void writeCollectively(const std::string &prefix, const int &rank, const int &manager_rank, MPI_Comm &comm_world);
	virtual void writeCollectively(const int &rank, const int &manager_rank, MPI_Comm &comm_world);
#endif

private :
	int id;
	std::string name;
	int group;
	long long starttime;
	std::string logprefix;

	// image name to log map.
	std::tr1::unordered_map<std::string, cci::common::LogSession > values;
};



}
}


#endif /* UTILS_LOGGER_H_ */
