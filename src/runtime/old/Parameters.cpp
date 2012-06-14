/*
 * Parameters.cpp
 *
 *  Created on: Apr 9, 2012
 *      Author: tcpan
 */

#include "Parameters.h"

namespace cci {

namespace runtime {


Parameters::~Parameters() {
	for (std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator iter = params.begin();
			iter != params.end(); ++iter) {
		iter->second.clear();
	}
	params.clear();
}

/**
 * insert has similar semantics as unordered_map insert - will replace existing.
 * if merge is desired, then use update function below.
 */
void Parameters::insert(const int workType, std::tr1::unordered_map<std::string, ValueBase &> &values) {
	params.insert(std::make_pair<int, std::tr1::unordered_map<std::string, ValueBase &> >(workType, values));
}
void Parameters::insert(const int workType, const std::string &name, ValueBase &value) {
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator pos =
		params.find(workType);
	if (pos == params.end()) {
		params.insert(std::make_pair<int, std::tr1::unordered_map<std::string, ValueBase &> >(workType, std::tr1::unordered_map<std::string, ValueBase &>()));
		pos = params.find(workType);
	}
	pos->second.insert(std::make_pair<std::string, ValueBase &>(name, value));
}

/**
 * erase has similar semantics as unordered_map erase
 * 1 if erased.  0 if not present.
 */
int Parameters::erase(const int workType) {
	return params.erase(workType);
}
int Parameters::erase(const int workType, const std::string &name) {
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator pos =
		params.find(workType);
	if (pos == params.end()) return 0;
	return pos->second.erase(name);
}
/**
 * find has similar semantics as unordered_map find
 */
std::tr1::unordered_map<std::string, ValueBase &> *Parameters::find(const int workType) {
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator pos =
		params.find(workType);
	if (pos == params.end()) return NULL;
	return &(pos->second);
}
ValueBase *Parameters::find(const int workType, const std::string &name) {
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator pos =
		params.find(workType);
	if (pos == params.end()) return NULL;
	std::tr1::unordered_map<std::string, ValueBase &>::iterator pos2 = pos->second.find(name);
	if (pos2 == set->end()) return NULL;
	return &(pos2->second);
}

// update the current object with values from the input
void Parameters::update(const Parameter *param) {
	for (std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator iter = param.begin();
			iter != param.end(); ++iter) {
		this->update(iter->first, iter->second);
	}
}
void Parameters::update(const int workType, const Parameter *param) {
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator pos =
		param->find(workType);
	if (pos == param->end()) return;
	this->update(workType, &(pos->second));
}
void Parameters::update(const int workType, std::tr1::unordered_map<std::string, ValueBase &> &values) {
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> >::iterator pos =
		params.find(workType);
	if (pos == params.end()) {
		// not in target.  so insert it.;
		params.insert(std::make_pair<int, std::tr1::unordered_map<std::string, ValueBase &> >(workType, values));
		return;
	}
	pos->second.insert(values.begin(), values.end());
}



}

}
