#include "profiler.h"
//#include <time.h>
#include <sys/time.h>


Profiler g_profiler;

Profiler::Profiler()
{
	m_enabled = false;
	m_fileUsed = false;	

	long long proc_freq;

//	::QueryPerformanceFrequency(&proc_freq);
//	m_frequency = proc_freq.QuadPart;	
}

bool Profiler::CheckAndCreateFile()
{	
	if(m_enabled == false) return false;
	if(m_fileUsed == true) return true;
	m_file.open("profilerInfo.txt");
	if(!m_file) return false;
	m_fileUsed = true;
	return true;
}

void Profiler::ClearTime(int id)
{
	if(m_time.size() <= id) {
		if(m_time.size() <= id) m_time.resize(id+1, 0);		
	}
	if(m_accumTime.size() <= id) {	
		if(m_accumTime.size() <= id) m_accumTime.resize(id+1, 0);
	}
	m_time[id] = 0;
	m_accumTime[id] = 0;
}

void Profiler::StartTime(int id, bool fromZero) {
	if(m_time.size() <= id) {		
		if(m_time.size() <= id) m_time.resize(id+1, 0);
	}
	if(m_accumTime.size() <= id) {
		if(m_accumTime.size() <= id) m_accumTime.resize(id+1, 0);
	}
	if(fromZero) m_accumTime[id] = 0;

	struct timeval ts;
	gettimeofday(&ts, NULL);
	m_time[id] = (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL; 

//	long long time;
//	::QueryPerformanceCounter(&time);	
//	m_time[id] = time.QuadPart;
}

void Profiler::EndTime(int id, char* outMsg)
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	long long tt = (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL; 
	long long newTime = m_accumTime[id] + (tt - m_time[id]);	
//	long long time;
//	::QueryPerformanceCounter(&time);	
//	long long newTime = m_accumTime[id] + (time.QuadPart - m_time[id]);	
	m_accumTime[id] = newTime;
//	if(outMsg != 0) PrintCustomTime(outMsg, 1000.0f*newTime/m_frequency);
	if(outMsg != 0) PrintCustomTime(outMsg, newTime);
}

void Profiler::PrintTime(int id, char* outMsg)
{
	if(outMsg != 0) PrintCustomTime(outMsg, m_accumTime[id]);
//	if(outMsg != 0) PrintCustomTime(outMsg, 1000.0f*m_accumTime[id]/m_frequency);
}

void Profiler::PrintCustomTime(char* outMsg, double time)
{
	if(!CheckAndCreateFile()) {;
		return;
	}
	m_file << outMsg << " " <<time<<std::endl;
}
