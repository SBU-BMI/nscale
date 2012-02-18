#ifndef _DLL_SBLA_H_
#define _DLL_SBLA_H_
#if defined SBLA_DLL_EXPORT
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR __declspec(dllimport)
#endif

#define LABELDATATYPE int
extern "C"{
	DECLDIR int LabelSBLA(LABELDATATYPE *data,int row,int col);
}
#endif