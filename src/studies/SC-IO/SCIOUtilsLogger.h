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


class SCIOLogSession {

public :

	SCIOLogSession() : id(-1), name(std::string()), session_name(std::string()), start(0LL) {};
	SCIOLogSession(const int &_id, const std::string &_name, const std::string &_session_name, long long &_start) :
		id(_id), name(_name), session_name(_session_name), events(), start(_start) {
		events.clear();
	};
	virtual ~SCIOLogSession() {
		events.clear();
	};

	virtual void log(cciutils::event e) {
		events.push_back(e);
	};

	virtual void toString(std::string &header, std::string &value) {
		std::stringstream ss1, ss2;
		ss1 << id << "," << name << "," << session_name << ",";
		ss2 << id << "," << name << "," << session_name << ",";

		for (int i = 0; i < events.size(); ++i) {
			ss1 << events[i].getName() << "," << events[i].getType() << ",";
			ss2 << (events[i].getSart() - start) << "," << (events[i].getEnd() - start) << ",";
		}
		header.assign(ss1.str());
		value.assign(ss2.str());
	};

	void setId(const int &_id) { id = _id; };
	void setName(const std::string &_name) { name.assign(_name); };
	void setSessionName(const std::string &_session_name) { session_name.assign(_session_name); };

private :
	int id;
	std::string name;
	std::string session_name;
	std::vector<cciutils::event> events;
	long long start;
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
	
	// session id is something like a filename or image name that is being processed.
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

private :
	int id;
	std::string name;
	long long starttime;

	// image name to log map.
	std::tr1::unordered_map<std::string, cciutils::SCIOLogSession > values;
};



}




#endif /* UTILS_LOGGER_H_ */
