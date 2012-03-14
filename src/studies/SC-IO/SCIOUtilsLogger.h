/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef SCIO_UTILS_LOGGER_H_
#define SCIO_UTILS_LOGGER_H_

#include <fstream>
#include <iostream>
#include <sys/time.h>

#include "utils.h"
#include <tr1/unordered_map>


namespace cciutils {

class event {
public:
	static const int COMPUTE = 0;
	static const int MEM_IO = 1;
	static const int FILE_IO = 2;
	static const int NETWORK_IO = 3;
	static const int OTHER = -1;

	event(const int &_id, const std::string &_name, const long long &_start, const long long &_end, const std::string &_annotation,
			const int &_type = -1) :
				id(_id), name(_name), starttime(_start), endtime(_end), annotation(_annotation), eventtype(_type) {};
	event(const int &_id, const std::string &_name, const long long &_start, const std::string &_annotation,
			const int &_type = -1) :
				id(_id), name(_name), starttime(_start), endtime(-1), annotation(_annotation), eventtype(_type) {};
	virtual ~event() {};

	virtual std::string getAsString() const {
		std::stringstream ss;
		ss << "[" << id << "]" << eventtype << "=\t" << name << ":\t" << starttime;
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
	long long getSart() const {
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


// having event logging does not get us to a default imple that consists of no-ops
// either caller need to construct events and manage their lifetime,
// or the default logger needs to delete the new'ed events.

// also because of c++ does not support templated virtual functions the whole thing
// becomes complicated - introduction of events, etc.

// abandom this approach.

//// this class exists because c++ does not support templated virtual functions.
//class EventBase {
//public :
//	EventBase(const char* en) {
//		eventName = en;
//	}
//	virtual ~EventBase() {};
//
//	virtual void printName(std::ostream &os) {
//		os << eventName;
//	}
//
//	virtual void printValue(std::ostream &os) {};
//
//	const char *eventName;
//
//};
//
//
//template <typename T>
//class Event : EventBase {
//public :
//	Event(const char* en, T &val) : EventBase(en) {
//		eventValue = val;
//	}
//	virtual ~Event() {};
//
//	virtual void printValue(std::ostream &os) {
//		os << eventValue;
//	}
//
//	T eventValue;
//};
//
//class Logger {
//public :
//
//	Logger() {};
//	virtual ~Logger() {};
//
//	virtual void endSession() {};
//
//	virtual void log(::cciutils::EventBase *ev) {};
//	virtual void setStart(::cciutils::EventBase *ev) {};
//	virtual void logTimeSinceLastStart(::cciutils::EventBase * ev) {};
//	virtual void logTimeSinceLastLog(::cciutils::EventBase * ev) {};
//
//	virtual void setT0(::cciutils::EventBase * ev) {};
//	virtual void logTimeSinceT0(::cciutils::EventBase * ev) {};
//
//	virtual void off() {};
//	virtual void on() {};
//	virtual void consoleOff() {};
//	virtual void consoleOn() {};
//
//};



/**
 *
 * holds on to the events in memory for multiple runs.  at the end, get all the data as a set of strings.
 *
 */
class SCIOLogger {
public :

	// _id is something like mpi rank or hostname
	SCIOLogger(const int &_id, const std::string &_name) :
		id(_id), name(_name) {};
	virtual ~SCIOLogger() {
	};
	
	// session id is something like a filename or image name that is being processed.
	virtual bool addSession(const std::string &session_name) {
		if (values.find(session_name) == values.end()) {
			std::vector<cciutils::event> vals;
			values[session_name] = vals;
		}
		return true;
	}

	virtual void endSession(const std::string &session_name) {}

	virtual void log(const std::string &session_name, cciutils::event &_event) {
		std::string eventName = _event.getName();
		int type = _event.getType();
		int eventId = _event.getId();

		if (headers.size() < eventId + 1) {
			headers.resize(eventId * 2);
			types.resize(eventId * 2);
		}
		if (eventName.compare(headers[eventId]) != 0) headers[eventId] = eventName;
		if (type != types[eventId]) types[eventId] = type;

		values[session_name] = _event;
	};

	virtual std::vector<std::string> toStrings() {
		// headers
		std::stringstream ss1, ss2, ss3;
		ss1 << ",,,";
		ss2 << ",,,";
		for (int i = 0; i < headers.size(); ++i) {
			ss1 << headers[i] << " start," << headers[i] << " end,";
			ss2 << types[i] << "," << types[i] << ",";
		}

		std::vector<std::string> output;
		output.push_back(ss1.str());
		output.push_back(ss2.str());

		ss1.str(std::string());
		ss2.str(std::string());

		for (std::tr1::unordered_map<std::string, std::vector<cciutils::event> >::iterator iter = values.begin();
				iter != values.end(); ++iter) {
			ss3 << id << "," << name << ",";

			ss3 << iter->first << ",";

			for (std::vector<cciutils::event>::iterator eiter = iter->second.begin(); eiter != iter->second.end(); ++eiter) {
				ss3 << eiter->getSart() << "," << eiter->getEnd() << ",";
			}

			output.push_back(ss3.str());

			ss3.str(std::string());
		}
		return output;
	}

private :
	std::string id;
	std::string name;
	std::vector<std::string> headers;
	std::vector<int> types;

	// image name to log map.
	std::tr1::unordered_map<std::string, std::vector<cciutils::event> > values;
};



}




#endif /* UTILS_LOGGER_H_ */
