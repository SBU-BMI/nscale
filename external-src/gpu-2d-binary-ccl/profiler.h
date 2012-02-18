#ifndef _UTILS_LIB_PROFILER_H_
#define _UTILS_LIB_PROFILER_H_


#include <vector>
#include <fstream>
#include <string>
//#include <windows.h>

class Profiler
{
public:
	Profiler();
	void SetEnabled(bool val) { m_enabled = val; }
	bool IsEnabled() const { return m_enabled; }
	void ClearTime(int id);
	void StartTime(int id, bool fromZero = true);

	void EndTime(int id, char* outMsg = 0);
	void PrintTime(int id, char* outMsg);

	void PrintCustomTime(char* outMsg, double time);
protected:
	bool CheckAndCreateFile();
private:
	std::vector<long long> m_time;
	std::vector<long long> m_accumTime;
	std::ofstream m_file;
	bool m_fileUsed;
	bool m_enabled;
	double m_frequency;
};

extern Profiler g_profiler;

#endif
