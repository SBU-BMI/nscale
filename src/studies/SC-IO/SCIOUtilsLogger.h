/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef SCIO_UTILS_LOGGER_H_
#define SCIO_UTILS_LOGGER_H_

#include <fstream>
#include <sys/time.h>
#include <sstream>
#include <tr1/unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <string.h>
#include <cstdlib>

#if defined (WITH_MPI)
#include "mpi.h"
#endif


namespace cciutils {

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
	static const int ADIOS_FINALIZE;
	static const int OTHER;

	event(const int &_id, const std::string &_name, const long long &_start, const long long &_end, const std::string &_annotation,
			const int &_type = -1) :
				id(_id), name(_name), starttime(_start), endtime(_end), annotation(_annotation), eventtype(_type) {};
	event(const int &_id, const std::string &_name, const long long &_start, const std::string &_annotation,
			const int &_type = -1) :
				id(_id), name(_name), starttime(_start), endtime(-1), annotation(_annotation), eventtype(_type) {};
	virtual ~event() {};

	virtual std::string getAsString() const {
		std::stringstream ss;
		ss << std::fixed << "[" << id << "]" << eventtype << "=\t" << name << ":\t" << starttime;
		if (endtime != -1)
			ss << "\t-\t" << endtime << "\t=\t" << (endtime - starttime);
		else
			ss << "\t\t\t\t";
		ss << "\t" << annotation;
		return ss.str();
	}

	virtual std::string getAsStringByType(const int &_type) const {
		if (eventtype == _type) {
			return getAsString();
		}
	}
	virtual std::string getAsStringByName(const std::string &_name) const {
		if (name.compare(_name) == 0) {
			return getAsString();
		}
	}
	virtual std::string getAsStringById(const int &_id) const {
		if (id == _id) {
			return getAsString();
		}
	}
	virtual std::string getName() const {
		return name;
	}
	int getType() const {
		return eventtype;
	}
	int getId() const {
		return id;
	}
	long long getStart() const {
		return starttime;
	}
	long long getEnd() const {
		return endtime;
	}
	void setEnd(const long long &_end) {
		endtime = _end;
	}

	static inline long long timestampInUS()
	{
		struct timeval ts;
		gettimeofday(&ts, NULL);
		return (ts.tv_sec*1000000LL + (ts.tv_usec));
		 //   timespec ts;
		//    clock_gettime(CLOCK_REALTIME, &ts);
		//    return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
	}


private:
	int id;
	std::string name;
	long long starttime;
	long long endtime;
	int eventtype;
	std::string annotation;
};



class SCIOLogSession {

public :

	SCIOLogSession() : id(-1), name(std::string()), session_name(std::string()), start(0LL) {};
	SCIOLogSession(const int &_id, const std::string &_name,
			const std::string &_session_name, long long &_start) :
		id(_id), name(_name), session_name(_session_name), events(), start(_start) {
		events.clear();

		countByEventName.clear();
		sumDurationByEventName.clear();
		sumSquareDurationByEventName.clear();;

		countByEventType.clear();
		sumDurationByEventType.clear();
		sumSquareDurationByEventType.clear();;
	};
	virtual ~SCIOLogSession() {
		events.clear();
		countByEventName.clear();
		sumDurationByEventName.clear();
		sumSquareDurationByEventName.clear();;

		countByEventType.clear();
		sumDurationByEventType.clear();
		sumSquareDurationByEventType.clear();;
	};
	virtual void restart() {
		events.clear();
		countByEventName.clear();
		sumDurationByEventName.clear();
		sumSquareDurationByEventName.clear();;

		countByEventType.clear();
		sumDurationByEventType.clear();
		sumSquareDurationByEventType.clear();;
	}
	virtual void log(cciutils::event e) {
		events.push_back(e);

		std::string ename = e.getName();
		int etype = e.getType();
		long long duration = e.getEnd() - e.getStart();
		countByEventName[ename] += 1;
		sumDurationByEventName[ename] += duration;
		sumSquareDurationByEventName[ename] += duration * duration;
		countByEventType[etype] += 1;
		sumDurationByEventType[etype] += duration;
		sumSquareDurationByEventType[etype] += duration * duration;
	};

	virtual void toString(std::string &header, std::string &value) {
		std::stringstream ss1, ss2;
		ss1 << "pid,hostName,sessionName,";
		ss2 << id << "," << name << "," << session_name << "," << std::fixed;

		for (int i = 0; i < events.size(); ++i) {
			ss1 << events[i].getName() << "," << events[i].getType() << ",";
			ss2 << (events[i].getStart() - start) << "," << (events[i].getEnd() - start) << ",";
		}
		header.assign(ss1.str());
		value.assign(ss2.str());
	};

	virtual void toOneLineString(std::string &value) {
		std::stringstream ss1;
		ss1 << "pid," << id << ",hostName," << name << ",sessionName," << session_name << "," << std::fixed;

		for (int i = 0; i < events.size(); ++i) {
			ss1 << events[i].getName() << "," << events[i].getType() << "," << (events[i].getStart() - start) << "," << (events[i].getEnd() - start) << ",";
		}
		value.assign(ss1.str());
	};

	virtual long getCountByEventName(std::string event_name) {
		return countByEventName[event_name];
	}
	virtual double getMeanByEventName(std::string event_name) {
		return double(sumDurationByEventName[event_name]) / double(countByEventName[event_name]);
	}
	virtual double getStdevByEventName(std::string event_name) {
		double mean = getMeanByEventName(event_name);
		long count =  countByEventName[event_name];
		if (count == 1) return -1;  //stdev for sample of 1 is undefined.
		return sqrt(double(sumSquareDurationByEventName[event_name]) / double(count) - mean * mean);
	}
	virtual long getCountByEventType(int event_type) {
		return countByEventType[event_type];
	}
	virtual double getMeanByEventType(int event_type) {
		return double(sumDurationByEventType[event_type]) / double(countByEventType[event_type]);
	}
	virtual double getStdevByEventType(int event_type) {
		double mean = getMeanByEventType(event_type);
		long count =  countByEventType[event_type];
		if (count == 1) return -1;  //stdev for sample of 1 is undefined.
		return sqrt(double(sumSquareDurationByEventType[event_type]) / double(count) - mean * mean);
	}

	virtual void toSummaryStringByName(std::string &header, std::string &value) {
		std::stringstream ss1, ss2;
		ss1 << "pid,hostName,sessionName,";
		ss2 << id << "," << name << "," << session_name << "," << std::fixed;

		for (std::tr1::unordered_map<std::string, long>::iterator iter= countByEventName.begin();
				iter != countByEventName.end(); ++iter) {
			std::string name = iter->first;
			ss1 << name << " count," << name << " mean," << name << " stdev, ";
			ss2 << iter->second << "," << getMeanByEventName(name) << "," << getStdevByEventName(name) << ",";
		}
		header.assign(ss1.str());
		value.assign(ss2.str());
	}
	virtual void toSummaryStringByType(std::string &header, std::string &value) {
		std::stringstream ss1, ss2;
		ss1 << "pid,hostName,sessionName,";
		ss2 << id << "," << name << "," << session_name << "," << std::fixed;

		for (std::tr1::unordered_map<int, long>::iterator iter = countByEventType.begin();
				iter != countByEventType.end(); ++iter) {
			int type = iter->first;
			ss1 << "type " << type << " count," << "type " << type << " mean," << "type " << type << " stdev, ";
			ss2 << iter->second << "," << getMeanByEventType(type) << "," << getStdevByEventType(type) << ",";
		}
		header.assign(ss1.str());
		value.assign(ss2.str());
	}

private :
	int id;
	std::string name;
	std::string session_name;
	std::vector<cciutils::event> events;
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
class SCIOLogger {
public :

	// _id is something like mpi rank or hostname
	SCIOLogger(const int &_id, const std::string &_name) :
		id(_id), name(_name) {
		starttime = cciutils::event::timestampInUS();	
	};

	virtual ~SCIOLogger() {
		values.clear();
	};
	
	// session id is something like a filename or image name or hostname.
	virtual cciutils::SCIOLogSession* getSession(const std::string &session_name) {
		if (values.find(session_name) == values.end()) {
			cciutils::SCIOLogSession session(id, name, session_name, starttime);
			values[session_name] = session;
		}
		return &(values[session_name]);
	}

	virtual std::vector<std::string> toStrings() {
		// headers
		std::vector<std::string> output;

		for (std::tr1::unordered_map<std::string, cciutils::SCIOLogSession >::iterator iter = values.begin();
				iter != values.end(); ++iter) {
			std::string headers;
			std::string times;

			iter->second.toString(headers, times);

			output.push_back(headers);
			output.push_back(times);
		}
		return output;
	}
	virtual std::vector<std::string> toOneLineStrings() {
		// headers
		std::vector<std::string> output;

		for (std::tr1::unordered_map<std::string, cciutils::SCIOLogSession >::iterator iter = values.begin();
				iter != values.end(); ++iter) {
			std::string times;

			iter->second.toOneLineString(times);

			output.push_back(times);
		}
		return output;
	}
	virtual std::vector<std::string> toSummaryStringsByName() {
		// headers
		std::vector<std::string> output;

		for (std::tr1::unordered_map<std::string, cciutils::SCIOLogSession >::iterator iter = values.begin();
				iter != values.end(); ++iter) {
			std::string headers;
			std::string times;

			iter->second.toSummaryStringByName(headers, times);

			output.push_back(headers);
			output.push_back(times);
		}
		return output;
	}
	virtual std::vector<std::string> toSummaryStringsByType() {
		// headers
		std::vector<std::string> output;

		for (std::tr1::unordered_map<std::string, cciutils::SCIOLogSession >::iterator iter = values.begin();
				iter != values.end(); ++iter) {
			std::string headers;
			std::string times;

			iter->second.toSummaryStringByType(headers, times);

			output.push_back(headers);
			output.push_back(times);
		}
		return output;
	}
	virtual void write(const std::string &prefix) {
	        std::vector<std::string> timings = this->toOneLineStrings();
        	std::stringstream ss;
	        for (int i = 0; i < timings.size(); i++) {
        	        ss << timings[i] << std::endl;
	        }
 
		std::stringstream fss;
		fss << prefix << "-" << id << ".csv";

		std::ofstream ofs2(fss.str().c_str());
        	ofs2 << ss.str() << std::endl;
	        ofs2.close();
	}

#if defined (WITH_MPI)
	virtual void writeCollectively(const std::string &prefix, const int &rank, const int &manager_rank, MPI_Comm &comm_world) {
	int size;
	MPI_Comm_size(comm_world, &size);


        // now do a collective io for the log
        std::vector<std::string> timings = this->toOneLineStrings();
        std::stringstream ss;
        for (int i = 0; i < timings.size(); i++) {
                ss << timings[i] << std::endl;
        }
        std::string logstr = ss.str();
        int logsize = logstr.size();

        char *sendlog = (char *)malloc(sizeof(char) * logsize + 1);
        memset(sendlog, 0, sizeof(char) * logsize + 1);
        strncpy(sendlog, logstr.c_str(), logsize);
        ss.str(std::string());

        int *recbuf = NULL;

	if (rank == manager_rank)
		recbuf = (int *) malloc(size * sizeof(int));
	
        // now send the thing to manager
        //      first gather sizes
        MPI_Gather(&logsize, 1, MPI_INT, recbuf, 1, MPI_INT, manager_rank, comm_world);


        //      then gatherv the messages.
        char *logdata = NULL;
        int * displbuf = NULL;
 

        if (rank == manager_rank) {
		// then perform exclusive prefix sum to get the displacement and the total length
		displbuf = (int *) malloc(size * sizeof(int));
		displbuf[0] = 0;
        	for (int i = 1; i < size; i++) {
                	displbuf[i] = displbuf[i-1] + recbuf[i-1];
	        }
        	int logtotalsize = displbuf[size - 1] + recbuf[size - 1];

	        logdata = (char*) malloc(logtotalsize * sizeof(char) + 1);
        	memset(logdata, 0, logtotalsize * sizeof(char) + 1);

	}		
        MPI_Gatherv(sendlog, logsize, MPI_CHAR, logdata, recbuf, displbuf, MPI_CHAR, manager_rank, comm_world);


        free(sendlog);

	if (rank == manager_rank) {
        	free(recbuf);
	        free(displbuf);
        
		std::stringstream fss;
		fss << prefix << ".csv";

		std::ofstream ofs2(fss.str().c_str());
        	ofs2 << logdata << std::endl;
	        ofs2.close();

        	//printf("%s\n", logdata);
		free(logdata);
	}
}
#endif

private :
	int id;
	std::string name;
	long long starttime;

	// image name to log map.
	std::tr1::unordered_map<std::string, cciutils::SCIOLogSession > values;
};



}




#endif /* UTILS_LOGGER_H_ */
