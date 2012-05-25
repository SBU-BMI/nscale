/*
 * Parameters.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tcpan
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <tr1/unordered_map>
#include <typeinfo>

namespace cci {

namespace runtime {

// base class.  should not be instantiable
class ValueBase {
public:
	ValueBase() {};
	virtual ~ValueBase() {};

	void getValue() = 0;
	virtual int getType() = 0;
	virtual int getSize() = 0;
	virtual unsigned char* getBytes() = 0;
};

template <typename T> struct type2id {};
template <int N> struct id2type {};

template <> struct type2id<unsigned char> { static int id = 1; };
template <> struct id2type<1> { typedef unsigned char type; };
template <> struct type2id<char> { static int id = 2; };
template <> struct id2type<2> { typedef char type; };
template <> struct type2id<unsigned short> { static int id = 3; };
template <> struct type2id<unsigned short int> { static int id = 3; };
template <> struct id2type<3> { typedef unsigned short type; };
template <> struct type2id<short> { static int id = 4; };
template <> struct type2id<short int> { static int id = 4; };
template <> struct id2type<4> { typedef short type; };
template <> struct type2id<unsigned int> { static int id = 5; };
template <> struct id2type<5> { typedef unsigned int type; };
template <> struct type2id<int> { static int id = 6; };
template <> struct id2type<6> { typedef int type; };
template <> struct type2id<unsigned long> { static int id = 7; };
template <> struct type2id<unsigned long int> { static int id = 7; };
template <> struct id2type<7> { typedef unsigned long type; };
template <> struct type2id<long> { static int id = 8; };
template <> struct type2id<long int> { static int id = 8; };
template <> struct id2type<8> { typedef long type; };
template <> struct type2id<bool> { static int id = 9; };
template <> struct id2type<9> { typedef bool type; };
template <> struct type2id<float> { static int id = 10; };
template <> struct id2type<10> { typedef float type; };
template <> struct type2id<double> { static int id = 11; };
template <> struct id2type<11> { typedef double type; };
template <> struct type2id<long double> { static int id = 12; };
template <> struct id2type<12> { typedef long double type; };
template <> struct type2id<wchar_t> { static int id = 13; };
template <> struct id2type<13> { typedef wchar_t type; };
template <> struct type2id<std::string> { static int id = 14; };
template <> struct id2type<14> { typedef std::string type; };


// typed base class.
template <typename T>
class Value : public ValueBase {
public:
	typedef T type;

	Value(T _val) : val(_val) {};
	Value(unsigned char * _bytes) {
		memcpy(bytes, _bytes, sizeof(T));
	}
	virtual ~Value() {};

	T getValue() { return val; };
	virtual int getType() { return type2id<T>::id; };
	virtual int getSize() { return sizeof(T); };
	virtual unsigned char* getBytes() { return bytes; };
private:
	union {
		T val;
		unsigned char bytes[sizeof(T)];
	};
};



class ValueFactory {
public:
	static ValueBase *createValue(const int _type, const int _size, const unsigned char* _bytes) {
		ValueBase * res = NULL;
		switch (_type) {

		case 1:
			res = new Value< id2type<1> >(_bytes);
			break;
		case 2:
			res = new Value< id2type<2> >(_bytes);
			break;
		case 3:
			res = new Value< id2type<3> >(_bytes);
			break;
		case 4:
			res = new Value< id2type<4> >(_bytes);
			break;
		case 5:
			res = new Value< id2type<5> >(_bytes);
			break;
		case 6:
			res = new Value< id2type<6> >(_bytes);
			break;
		case 7:
			res = new Value< id2type<7> >(_bytes);
			break;
		case 8:
			res = new Value< id2type<8> >(_bytes);
			break;
		case 9:
			res = new Value< id2type<9> >(_bytes);
			break;
		case 10:
			res = new Value< id2type<10> >(_bytes);
			break;
		case 11:
			res = new Value< id2type<11> >(_bytes);
			break;
		case 12:
			res = new Value< id2type<12> >(_bytes);
			break;
		case 13:
			res = new Value< id2type<13> >(_bytes, _size);
			break;
		case 14:
			res = new Value< id2type<14> >(_bytes);
			break;
		default:
			break;
		}

		return res;
	}
};


/**
 *  parameter for a work object.
 *
 *  use this to track output from multiple paths in DAG
 *  specifically, use an global unique id to identify
 *  and to merge later.
 *
 *	this approach allows "update" when the workers are
 *	in communication with their manager.
 *
 *  also contain the work type id.  This is used by the work object
 *  to assess if the required parameters are all set.
 *
 *  parameters then is just a container of parameters.
 *
 */
class Parameters {

	friend int ParameterSerialization::serialize(const Parameters &params, void* &output, int &output_size);
	friend int ParameterSerialization::deserialize(const void* input, const int input_size, Parameters &params);

public:
	// globally unique id.  should be unique across machine and processes.
	// for example, concatenate the rank of the machine with a sequential id.
	uint64_t id;

	Parameters() {};
	~Parameters();

	/**
	 * insert has similar semantics as unordered_map insert
	 * this means reinsersion will replace.  If merge is wanted, use update.
	 */
	void insert(const int workType, std::tr1::unordered_map<std::string, ValueBase &> &values);
	void insert(const int workType, const std::string &name, ValueBase &value);

	/**
	 * erase has similar semantics as unordered_map erase
	 */
	int erase(const int workType);
	int erase(const int workType, const std::string &name);
	/**
	 * find has similar semantics as unordered_map find
	 */
	std::tr1::unordered_map<std::string, ValueBase &> *find(const int workType);
	ValueBase *find(const int workType, const std::string &name);

	// update the current object with values from the input
	int update(const Parameter *param);
	int update(const int workType, const Parameter *param);
	int update(const int workType, std::tr1::unordered_map<std::string, ValueBase &> &values);

private:
	// hash table of workType to work params, which is then hashtable of name - value
	std::tr1::unordered_map<int, std::tr1::unordered_map<std::string, ValueBase &> > params;

};

}

}

#endif /* PARAMETERS_H_ */
