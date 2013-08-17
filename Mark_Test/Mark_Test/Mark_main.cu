#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <ctime>
#include <cuda.h>
#include <math_functions.h>
//not needed now, but will use in future
#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
using namespace std;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice
#define BLOCK_SIZE 16
#define BLOCKSIZE BLOCK_SIZE
#define THREADS 256
#define LINEAR_BLOCK_SIZE THREADS
#define DO_TEST 1
#define ELEM_ESTIMATE 230422
#define NUM_INPUTS 12
#define NUM_BASE 21
#define PATTERN_SIZE (1<<NUM_INPUTS)
#define MP make_pair

typedef pair<int,int> Pii;
typedef pair<int,Pii> Piii;
typedef pair<float,float> Pff;
typedef pair<Pff,Pii> PFI;

const float eps=0.00001f, SENTINEL=-11111.11111f,risk_free_rate=0.02f;
const float PTarg[NUM_BASE]={0.008f,0.0081f,0.0082f,0.0083f,0.0084f,0.0085f,0.0086f,0.0087f,0.0088f,0.0089f,0.009f,0.0091f,0.0092f,0.0093f,0.0094f,0.0095f,0.0096f,0.0097f,0.0098f,0.0099f,0.01f};
const float SLoss[NUM_BASE]={0.007f,0.0071f,0.0072f,0.0073f,0.0074f,0.0075f,0.0076f,0.0077f,0.0078f,0.0079f,0.008f,0.0081f,0.0082f,0.0083f,0.0084f,0.0085f,0.0086f,0.0087f,0.0088f,0.0089f,0.009f};
const float ThreshValsLong[NUM_BASE]={0.0002f,0.00021f,0.00022f,0.00023f,0.00024f,0.00025f,0.00026f,0.00027f,0.00028f,0.00029f,0.0003f,0.00031f,0.00032f,0.00033f,0.00034f,0.00035f,0.00036f,0.00037f,0.00038f,0.00039f,0.0004f};
const float ThreshValsShort[NUM_BASE]={0.001f,0.00102f,0.00104f,0.00106f,0.00108f,0.0011f,0.00112f,0.00114f,0.00116f,0.00118f,0.0012f,0.00122f,0.00124f,0.00126f,0.00128f,0.0013f,0.00132f,0.00134f,0.00136f,0.00138f,0.0014f};

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

//CPU inlines
inline bool _feq(float a){return ((a+0.0000001f)>=0.0f && (a-0.0000001f)<=0.0f);}
inline bool _eq(float a,float b){return ((a+0.0001f)>=b && (a-0.0001f)<=b);}
inline int cntbt(int n){n=n-((n>>1)&0x55555555);n=(n&0x33333333)+((n>>2)&0x33333333);return (((n+(n>>4))&0x0F0F0F0F)*0x01010101)>>24;}
inline int covertToBase(int a,int b, int c, int d){return (a*9261+b*441+c*21+d);}

//CPU function declarations
void converTfour(int num, int &a,int &b, int &c,int &d,const int range);
int read_in_data(const string name, float *O, float *H,float *L, float *C);
int get_state(const float *O, const float *H, const float *L, const float *C,const int idx, const int days_back,const int adj,const int adj2,float p0,float p1);
void cpu_fill_in(const float *O, const float *H, const float *L, const float *C,const int sz,int *StateArr,int *CPUfreq,float *ExpVal,
					const int days_back,const int days_forward,const int adj,const int adj2,float p0,float p1);

void cpu_scale(const int *CPUfreq,float *ExpVal,const int N);
Pii most_valuable_inputs(const float *expval, const int N);
void cpu_optimize(const float *O, const float *H, const float *L, const float *C,const int sz,int *StateArr,const int days_back,
					float *ExpVal,float &long_thres,float &short_thres,float &stop_loss, float &target,float &best,int &wnz, int &lz);


//GPU constant memory definition


__constant__ float D_SENTINEL=-11111.11111f;
__constant__ float D_PTarg[NUM_BASE]={0.008f,0.0081f,0.0082f,0.0083f,0.0084f,0.0085f,0.0086f,0.0087f,0.0088f,0.0089f,0.009f,0.0091f,0.0092f,0.0093f,0.0094f,0.0095f,0.0096f,0.0097f,0.0098f,0.0099f,0.01f};
__constant__ float D_SLoss[NUM_BASE]={0.007f,0.0071f,0.0072f,0.0073f,0.0074f,0.0075f,0.0076f,0.0077f,0.0078f,0.0079f,0.008f,0.0081f,0.0082f,0.0083f,0.0084f,0.0085f,0.0086f,0.0087f,0.0088f,0.0089f,0.009f};
__constant__ float D_ThreshValsLong[NUM_BASE]={0.0002f,0.00021f,0.00022f,0.00023f,0.00024f,0.00025f,0.00026f,0.00027f,0.00028f,0.00029f,0.0003f,0.00031f,0.00032f,0.00033f,0.00034f,0.00035f,0.00036f,0.00037f,0.00038f,0.00039f,0.0004f};
__constant__ float D_ThreshValsShort[NUM_BASE]={0.001f,0.00102f,0.00104f,0.00106f,0.00108f,0.0011f,0.00112f,0.00114f,0.00116f,0.00118f,0.0012f,0.00122f,0.00124f,0.00126f,0.00128f,0.0013f,0.00132f,0.00134f,0.00136f,0.00138f,0.0014f};
//GPU kernel declarations
__global__ void fill_in_state(const float *O, const float *H, const float *L, const float *C,int *StateArr,int *freq,
							float *ExpVal,const int sz,const int days_back,const int days_forward,const int adj,
							const int adj2,const float p0, const float p1);

__global__ void gpu_scale(const int *freq,float *ExpVal, const int N);

__global__ void gpu_optimize_threshold(const float *O, const float *H, const float *L, const float *C,const int *StateArr,const float *ExpVal,
										float *BlkBest, int *BaseNum,const int sz,const int range,const int days_back,const int bound);

///////////////////////////MAIN TESTING AND TIMING///////////////////////////////////////////////////////////////////////////////////////////////

int main(){
	char ch;
	const int N=PATTERN_SIZE,comboSpace=NUM_BASE*NUM_BASE*NUM_BASE*NUM_BASE;
	const int periods_back=9,lim=12,periods_forward=4;

	float *Open=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	float *High=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	float *Low=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	float *Close=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	int *PatternsCPU=(int *)malloc(ELEM_ESTIMATE*sizeof(int));
	int *PatternsGPU=(int *)malloc(ELEM_ESTIMATE*sizeof(int));
	int *CPUfreq=(int *)malloc(N*sizeof(int));
	int *GPUfreq=(int *)malloc(N*sizeof(int));
	float *CPUexpval=(float *)malloc(N*sizeof(float));
	float *GPUexpval=(float *)malloc(N*sizeof(float));

	string Pdata="Euro_2009-2012(5min).txt";
	cout<<"\nLoading Data..\n";
	const int num_elements=read_in_data(Pdata,Open,High,Low,Close);
	if(num_elements<=periods_back){
		cout<<"Error in file.\n";
	}
	
	memset(PatternsCPU,-1,ELEM_ESTIMATE*sizeof(int));
	memset(PatternsGPU,-1,ELEM_ESTIMATE*sizeof(int));
	memset(CPUfreq,0,N*sizeof(int));
	memset(CPUexpval,0,N*sizeof(int));

	float CPUans=0.0f,h_lt=0.0f,h_st=0.0f,h_sl=0.0f,h_t=0.0f;
	float GPUans=0.0f,d_lt=0.0f,d_st=0.0f,d_sl=0.0f,d_t=0.0f;

	int CPUcombo=0,GPUcombo=0,h_w=0,h_l=0;
	cout<<"Starting CPU testing:\n";
	UINT wTimerRes = 0;

	bool init = InitMMTimer(wTimerRes);
	DWORD startTime = timeGetTime(),GPUtime=0,CPUtime=0;

	cpu_fill_in(Open,High,Low,Close,num_elements,PatternsCPU,CPUfreq,CPUexpval,periods_back,periods_forward,5,6,1.0f,1.0f);
	cpu_scale(CPUfreq,CPUexpval,N);
	cpu_optimize(Open,High,Low,Close,num_elements,PatternsCPU,periods_back,CPUexpval,h_lt,h_st,h_sl,h_t,CPUans,h_w,h_l);

	DWORD endTime = timeGetTime();
	CPUtime=endTime-startTime;
	cout<<"CPU timing(fill only): "<<CPUtime<< " ms.\n";
	DestroyMMTimer(wTimerRes, init);

	

	cout<<"CPU:\n";
	cout<<"\nThe best profit possible after parameter optimization was "<<10000.0f*CPUans<<"ticks.\n";
	cout<<"\nThe parameters associated with the best-case return were: \n";
	cout<<"Long threshold = "<< h_lt*100.0f<<"% , Short threshold = "<<h_st*100.0f<<"% .\n";
	cout<<"Stop loss = "<<h_sl*100.0f<<"%, and profit target = "<<h_t*100.0f<<"% .\n\n";


	if(DO_TEST){
		cout<<"Starting GPU testing:\n";
		float *O,*H,*L,*C,*ExpVal,*BlkBest;
		int *Patterns,*freq,*BaseNum;
		const int num_bytes_data=num_elements*sizeof(float);
		const int num_blx=((comboSpace+THREADS-1)/THREADS);
	
		cudaError_t err=cudaDeviceReset();//not really needed, but will reset device
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMalloc((void **)&O,num_bytes_data);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&H,num_bytes_data);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&L,num_bytes_data);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&C,num_bytes_data);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&Patterns,num_elements*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&freq,N*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&ExpVal,N*sizeof(float));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&BlkBest,num_blx*sizeof(float));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&BaseNum,num_blx*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemset(ExpVal,0,N*sizeof(float));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(freq,0,N*sizeof(float));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		//copy data to GPU
		err=cudaMemcpy(O,Open,num_bytes_data,_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(H,High,num_bytes_data,_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(L,Low,num_bytes_data,_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(C,Close,num_bytes_data,_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(Patterns,PatternsGPU,num_elements*sizeof(int),_HTD);//not really needed, did for error checking
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		

		const int num_blocks=(num_elements+THREADS-1)/THREADS;//for fill step only

		thrust::device_ptr<float> d_p=thrust::device_pointer_cast(BlkBest);

		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();

		fill_in_state<<<num_blocks,THREADS>>>(O,H,L,C,Patterns,freq,ExpVal,num_elements,periods_back,periods_forward,5,6,1.0f,1.0f);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		gpu_scale<<<((N+THREADS-1)/THREADS),THREADS>>>(freq,ExpVal,N);

		gpu_optimize_threshold<<<num_blx,THREADS>>>(O,H,L,C,Patterns,ExpVal,BlkBest,BaseNum,num_elements,NUM_BASE,periods_back,comboSpace);

		thrust::device_ptr<float> result=thrust::max_element(d_p,d_p+num_blx);

		GPUans= *result;

		unsigned long long index=unsigned long long(result-d_p);
		err=cudaMemcpy(&GPUcombo,BaseNum+int(index),sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		endTime = timeGetTime();
		GPUtime=endTime-startTime;
		cout<<"GPU timing: "<<GPUtime<<" ms.\n";
		DestroyMMTimer(wTimerRes, init);

		err=cudaMemcpy(PatternsGPU,Patterns,num_elements*sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(GPUfreq,freq,N*sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(GPUexpval,ExpVal,N*sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		//free GPU data
		err=cudaFree(O);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(H);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(L);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(C);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(Patterns);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(freq);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(ExpVal);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	int a,b,c,d;
	converTfour(GPUcombo,a,b,c,d,NUM_BASE);
	
	d_lt=ThreshValsLong[a];d_st=ThreshValsShort[b];
	d_sl=SLoss[c];d_t=PTarg[d];

	cout<<"GPU:\n";
	cout<<"\nThe best profit possible after parameter optimization was "<<10000.0f*GPUans<<"ticks.\n";
	cout<<"\nThe parameters associated with the best-case return were: \n";
	cout<<"Long threshold = "<< d_lt*100.0f<<"% , Short threshold = "<<d_st*100.0f<<"% .\n";
	cout<<"Stop loss = "<<d_sl*100.0f<<"%, and profit target = "<<d_t*100.0f<<"% .\n\n";

	cout<<"\nFor the entire fill, scale, and optimize steps the CUDA implementation was "<<float(CPUtime)/float(GPUtime)<<" faster than the CPU implementation.\n";
	


	free(Open);
	free(High);
	free(Low);
	free(Close);
	free(PatternsCPU);
	free(PatternsGPU);
	free(CPUfreq);
	free(GPUfreq);
	free(CPUexpval);
	free(GPUexpval);

	//MapExp.clear();
	cin>>ch;
	return 0;
	
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}

int read_in_data(const string name, float *O, float *H,float *L, float *C){
	int num_elements=0;
	FILE *file;
	file=fopen(name.c_str(),"r");
	int m,d,y,t,e0,e1;
	if(file!=NULL){
		while(!feof(file)){
			fscanf(file,"%d/%d/%d %d %f %f %f %f %d %d",&m,&d,&y,&t,&O[num_elements],&H[num_elements],&L[num_elements],&C[num_elements],&e0,&e1);
			num_elements++;
		}
	}else{
		cout<<"\nCould not find file "<<name<<" in current working directory!\n";
		return -1;
	}
	fclose(file);
	return num_elements;
}
void converTfour(int num, int &a,int &b, int &c,int &d,const int range){
	a=b=c=d=range-1;
	while((a*9261)>num){a--;}
	num-=(a*9261);
	while( (b*441)>num){b--;}
	num-=(b*441);
	while( (c*21)>num){c--;}
	num-=(c*21);
	d=num;
	return;
}
int get_state(const float *O, const float *H, const float *L, const float *C,const int idx, const int days_back,const int adj,const int adj2,float p0,float p1){
	assert(days_back>1);//otherwise may divide by zero
	assert(idx>=days_back);
	int ans=0,cc=0;
	float last=0.0f,avgr=0.0f;
	float full_mvg_avg=0.0f,half_mvg_avg=0.0f;
	for(int i=idx-days_back+1;i<=idx;i++){

		float range=(H[i]-L[i]);
		avgr+=range;

		float avg_p=((2.0f*C[i])+L[i]+H[i]+O[i])/5.0f;
		if(cc && (avg_p*p0)<(last*p1))ans|=(1<<cc);
		full_mvg_avg+=avg_p;
		if(cc>((days_back)>>1))half_mvg_avg+=avg_p;
		last=avg_p;
		cc++;
	}
	avgr/=float(days_back);
	full_mvg_avg/=float(days_back);
	half_mvg_avg/=float(((days_back)>>1));

	if(half_mvg_avg<full_mvg_avg)ans|=1;
	//set bits 10 through 11
	if((H[idx-adj]-L[idx-adj2])>avgr)ans|=1024;
	if(p0*(H[idx-adj2]-L[idx-adj])<(p1*avgr))ans|=2048;

	return ans;
}
void cpu_fill_in(const float *O, const float *H, const float *L, const float *C,const int sz,int *StateArr,int *CPUfreq,float *ExpVal,
	const int days_back,const int days_forward,const int adj,const int adj2,float p0,float p1){

	for(int i=days_back;(i+days_forward)<sz;i++){
		StateArr[i]=get_state(O,H,L,C,i,days_back,adj,adj2,p0,p1);
		CPUfreq[StateArr[i]]++;
		float t=0.0f;
		for(int j=0;j<days_forward;j++){
			t+=0.5f*(H[i+j+1]+L[i+j+1]);
		}
		assert(C[i]>0.0f);
		t=(C[i]-(t/float(days_forward)))/C[i];
		ExpVal[StateArr[i]]+=t;
	}
}
void cpu_scale(const int *CPUfreq,float *ExpVal,const int N){
	for(int i=0;i<N;i++){
		if(CPUfreq[i]>0){
			ExpVal[i]/=float(CPUfreq[i]);
		}
	}
}
Pii most_valuable_inputs(const float *expval, const int N){
	Pii ret=MP(-1,-1);
	int bull[NUM_INPUTS]={0},bear[NUM_INPUTS]={0};
	for(int i=0;i<N;i++){
		if(expval[i]>0.0f){
			for(int j=0;j<NUM_INPUTS;j++){
				if(i&(1<<j))bull[j]++;
			}
		}else if(expval[i]<0.0f){
			for(int j=0;j<NUM_INPUTS;j++){
				if(i&(1<<j))bear[j]++;
			}
		}
	}
	int gf=0,bf=0;
	for(int i=0;i<NUM_INPUTS;i++){
		if(bull[i]>gf){
			gf=bull[i];
			ret.first=i;
		}
		if(bear[i]>bf){
			bf=bear[i];
			ret.second=i;
		}
	}
	return ret;
}

void cpu_optimize(const float *O, const float *H, const float *L, const float *C,const int sz,int *StateArr,const int days_back,
	float *ExpVal,float &long_thres,float &short_thres,float &stop_loss, float &target,float &best,int &wnz, int &lz){

	best=-999.9f;
	float lt,st,sl,t,p,last,dif,cur;
	int dir,wins,losses;
	for(int ii=0;ii<NUM_BASE;ii++)for(int j=0;j<NUM_BASE;j++)for(int k=0;k<NUM_BASE;k++)for(int m=0;m<NUM_BASE;m++){

		lt=ThreshValsLong[ii];st=ThreshValsShort[j];sl=SLoss[k];t=PTarg[m];
		dir=0;wins=0;losses=0;
		p=0.0f;last=SENTINEL;dif=0.0f;
		for(int i=days_back;i<sz;i++){
			cur=ExpVal[StateArr[i]];
			if(dir!=0){//there is an open position
				if(cur>=lt){
					//if already long, stay in long position
					//else if short get out and get long
					if(dir==-1){
						dif=(last-C[i]);
						if(dif>0.0f)wins++;
						else
							losses++;
						p+=dif;
						last=C[i];
						dir=1;
					}
				}else if(cur<=(-st)){
					//if already short stay short
					//else if long get out and get short
					if(dir==1){
						dif=(C[i]-last);
						if(dif>0.0f)wins++;
						else
							losses++;
						p+=dif;
						last=C[i];
						dir=-1;
					}
				}else{//get out of long if beyond long/short profit threshold or have an open loss on position of over sl
					if(dir==-1){
						dif=(last-C[i]);
						if(dif>0.0f && (dif/last)>=t){//have reached profit targer from short position, get out now and get neutral
							wins++;
							p+=dif;
							dir=0;
							last=SENTINEL;
						}else if(dif<0.0f && C[i]>=(last+(last*sl))){//stop loss
							p+=dif;
							dir=0;
							last=SENTINEL;
							losses++;
						}
					}else if(dir==1){
						dif=(C[i]-last);
						if(dif>0.0f && (dif/last)>=t){//have reached profit targer from short position, get out now and get neutral
							wins++;
							p+=dif;
							dir=0;
							last=SENTINEL;
						}else if(dif<0.0f && C[i]<=(last-(last*sl))){//stop loss
							p+=dif;
							dir=0;
							last=SENTINEL;
							losses++;
						}	
					}
				}
			}else{//no current open position
				if(cur>=lt){//open a new long position
					last=C[i];
					dir=1;
					dif=0.0f;

				}else if(cur<=(-st)){//open a new short position
					last=C[i];
					dir=-1;
					dif=0.0f;
				}
			}
		}
		if(dir!=0){
			dif= (dir==-1) ? (last-C[sz-1]):(C[sz-1]-last);
			p+=dif;
		}
		if(p>best){//save best profit and the combo which gave that profit
			best=p;
			long_thres=lt;
			short_thres=-st;
			stop_loss=sl;
			target=t;
			wnz=wins;
			lz=losses;
		}
	}
}

__global__ void fill_in_state(const float *O, const float *H, const float *L, const float *C,int *StateArr,int *freq,
							float *ExpVal,const int sz,const int days_back,const int days_forward,const int adj,
							const int adj2,const float p0, const float p1){

	const int offset=blockIdx.x*blockDim.x + threadIdx.x;
	if((offset+days_forward)<sz && offset>=days_back){
		int ans=0,cc=0,daysb2=(days_back>>1);
		float db=float(days_back),full_mvg_avg=0.0f,half_mvg_avg=0.0f,avg_p,last=0.0f,avgr=0.0f;
		for(int i=offset-days_back+1;i<=offset;i++){
			avgr+=(H[i]-L[i]);
			avg_p=((2.0f*C[i])+L[i]+H[i]+O[i])/5.0f;
			if(cc && (avg_p*p0)<(last*p1))ans|=(1<<cc);
			full_mvg_avg+=avg_p;
			if(cc>daysb2)half_mvg_avg+=avg_p;
			last=avg_p;
			cc++;
		}
		avgr/=db;
		full_mvg_avg/=db;
		half_mvg_avg/=float(daysb2);

		if(half_mvg_avg<full_mvg_avg)ans|=1;
		//set bits 10 through 11
		if((H[offset-adj]-L[offset-adj2])>avgr)ans|=1024;
		if(p0*(H[offset-adj2]-L[offset-adj])<(p1*avgr))ans|=2048;
		//now add to freq of that state
		atomicAdd(&freq[ans],1);//use atomics to store frequency of that specific pattern

		StateArr[offset]=ans;//store the pattern for that time period

		avg_p=0.0f;

		#pragma unroll
		for(cc=0;cc<days_forward;cc++){
			avg_p+=0.5f*(H[offset+cc+1]+L[offset+cc+1]);
		}
		avg_p=(C[offset]-(avg_p/float(days_forward)))/C[offset];
		atomicAdd(&ExpVal[ans],avg_p);//keep track of the average price change given that particular pattern
	}
}

__global__ void gpu_scale(const int *freq,float *ExpVal, const int N){//this performs the final adjustment to determine the average price change following each distinct price/volatility pattern
	const int offset=blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<N){
		if(freq[offset]>0){
			ExpVal[offset]/=float(freq[offset]);
		}
	}
}

__global__ void gpu_optimize_threshold(const float *O, const float *H, const float *L, const float *C,const int *StateArr,const float *ExpVal,
										float *BlkBest, int *BaseNum,const int sz,const int range,const int days_back,const int bound){

	const int offset=blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ volatile float best[THREADS];//MUST BE VOLATILE!!!!!!!

	float t_best=-999999.9f;

	if(offset<bound){
		
		int num=offset,a,b,c,dir=0;
		a=b=c=(range-1);
		while((a*9261)>num){a--;}
		num-=(a*9261);
		while( (b*441)>num){b--;}
		num-=(b*441);
		while( (c*21)>num){c--;}
		num-=(c*21);
		float lt=D_ThreshValsLong[a],st=D_ThreshValsShort[b],sl=D_SLoss[c],t=D_PTarg[num],dif=0.0f,last=D_SENTINEL,cur;
		// now have the optimization parameters so test
		t_best=0.0f;
		for(int i=days_back;i<sz;i++){
			cur=ExpVal[StateArr[i]];
			if(dir!=0){//there is an open position
				if(cur>=lt){
					//if already long, stay in long position
					//else if short get out and get long
					if(dir==-1){
						dif=last-C[i];
						t_best+=dif;
						last=C[i];
						dir=1;
					}
				}else if(cur<=(-st)){
						//if already short stay short
						//else if long get out and get short
					if(dir==1){
						dif=C[i]-last;
						t_best+=dif;
						last=C[i];
						dir=-1;
					}
				}else{//get out of long if beyond long/short profit threshold or have an open loss on position of over sl
					if(dir==-1){
						dif=last-C[i];
						if(dif>0.0f && (dif/last)>=t){//have reached profit targer from short position, get out now and get neutral
							t_best+=dif;
							dir=0;
							last=D_SENTINEL;
						}else if(dif<0.0f && C[i]>=(last+(last*sl))){//stop loss
							t_best+=dif;
							dir=0;
							last=D_SENTINEL;
						}
					}else if(dir==1){
						dif=C[i]-last;
						if(dif>0.0f && (dif/last)>=t){//have reached profit targer from short position, get out now and get neutral
							t_best+=dif;
							dir=0;
							last=D_SENTINEL;
						}else if(dif<0.0f && C[i]<=(last-(last*sl))){//stop loss
							t_best+=dif;
							dir=0;
							last=D_SENTINEL;
						}	
					}
				}
			}else{//no current open position
				if(cur>=lt){//open a new long position
					last=C[i];
					dir=1;
					dif=0.0f;
				}else if(cur<=(-st)){//open a new short position
					last=C[i];
					dir=-1;
					dif=0.0;
				}
			}
		}
		if(dir!=0){//in case there is still an open position
			dif= (dir==-1) ? (last-C[sz-1]):(C[sz-1]-last);
			t_best+=dif;
		}
	}

	best[threadIdx.x]=t_best;
	__syncthreads();

	//assuming 256 THREADS, change if different
	if(threadIdx.x<128){
		best[threadIdx.x] = (best[threadIdx.x+128] > best[threadIdx.x]) ? best[threadIdx.x+128] : best[threadIdx.x];
	}
	__syncthreads();
	if(threadIdx.x<64){
		best[threadIdx.x] = (best[threadIdx.x+64] > best[threadIdx.x]) ? best[threadIdx.x+64] : best[threadIdx.x];
	}
	__syncthreads();
	if(threadIdx.x<32){
		best[threadIdx.x] = best[threadIdx.x+32] > best[threadIdx.x] ? best[threadIdx.x+32] : best[threadIdx.x];
		best[threadIdx.x] = best[threadIdx.x+16] > best[threadIdx.x] ? best[threadIdx.x+16] : best[threadIdx.x];
		best[threadIdx.x] = best[threadIdx.x+8] > best[threadIdx.x] ? best[threadIdx.x+8] : best[threadIdx.x];
		best[threadIdx.x] = best[threadIdx.x+4] > best[threadIdx.x] ? best[threadIdx.x+4] : best[threadIdx.x];
		best[threadIdx.x] = best[threadIdx.x+2] > best[threadIdx.x] ? best[threadIdx.x+2] : best[threadIdx.x];
		best[threadIdx.x] = best[threadIdx.x+1] > best[threadIdx.x] ? best[threadIdx.x+1] : best[threadIdx.x];	
	}
	__syncthreads();

	if(threadIdx.x==0){
		BlkBest[blockIdx.x]=best[0];
	}
	if(best[0]==t_best){
		BaseNum[blockIdx.x]=offset;
	}
}


