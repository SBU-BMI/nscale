/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef UTILS_LOGGER_H_
#define UTILS_LOGGER_H_

#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "utils.h"

namespace cciutils {

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


class SimpleCSVLogger {
public :

	SimpleCSVLogger(const char* name) {
		char headername[1024];
		char valuename[1024];
		strcpy(headername, name);
		strcat(headername, "-header.csv");
		strcpy(valuename, name);
		strcat(valuename, "-value.csv");
		header.open(headername, std::ios_base::out | std::ios_base::app);
		value.open(valuename, std::ios_base::out | std::ios_base::app);
		t0 = 0;
		start = 0;
		curr = 0;
		last = 0;
		_consoleOn = true;
		_on = true;
	};
	virtual ~SimpleCSVLogger() {
		header.flush();
		header.close();
		value.flush();
		value.close();
	};
	
	void endSession() {
		if (_on) {
			header << std::endl;
			value << std::endl;
			header.flush();
			value.flush();
		};
	}

	template <typename T>
	void log(const char *eventName, T eventValue) {
		if (_on) {
			header << eventName<< ", ";
			value << eventValue << ", ";
		}
		if (_consoleOn) {
			std::cout << "[LOGGER] " << eventName <<": " << eventValue << std::endl;
		}
	};

	void logStart(const char *eventName) {
		start = cciutils::ClockGetTime();
		log(eventName, start);
	}
	void logTimeSinceLastStart(const char *eventName) {
		curr = cciutils::ClockGetTime();
		uint64_t elapsed = curr - start;
		if (start == 0) elapsed = 0UL;
		log(eventName, elapsed);
	}

	void logTimeSinceLastLog(const char *eventName) {
		curr = cciutils::ClockGetTime();
		uint64_t elapsed = curr - last;
		if (last == 0) elapsed = 0;
		log(eventName, elapsed);
		last = curr;
	}

	void logT0(const char *eventName) {
		t0 = cciutils::ClockGetTime();
		log(eventName, t0);
	}
	void logTimeSinceT0(const char *eventName) {
		curr = cciutils::ClockGetTime();
		uint64_t elapsed = curr - t0;
		if (t0 == 0) elapsed = 0UL;
		log(eventName, elapsed);
	}

	void off() {
		_on = false;
	}
	void on() {
		_on = true;
	}
	void consoleOff() {
		_consoleOn = false;
	}
	void consoleOn() {
		_consoleOn = true;
	}
private :
	std::ofstream header;
	std::ofstream value;
	uint64_t t0;
	uint64_t start;

	uint64_t last;
	uint64_t curr;
	bool _on;
	bool _consoleOn;
};



}




#endif /* UTILS_LOGGER_H_ */
