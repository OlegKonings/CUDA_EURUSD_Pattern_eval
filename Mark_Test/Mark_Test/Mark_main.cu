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
#define PATTERN_SIZE (1<<NUM_INPUTS)
#define MP make_pair
#define PRIME 7919

typedef pair<int,int> Pii;
typedef pair<int,Pii> Piii;
typedef pair<float,float> Pff;
typedef pair<Pff,Pii> PFI;

const float eps=0.00001f, SENTINEL=-11111.11111f,risk_free_rate=0.02f;

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

inline bool _feq(float a){return ((a+0.0000001f)>=0.0f && (a-0.0000001f)<=0.0f);}
inline bool _eq(float a,float b){return ((a+0.0001f)>=b && (a-0.0001f)<=b);}
inline int cntbt(int n){n=n-((n>>1)&0x55555555);n=(n&0x33333333)+((n>>2)&0x33333333);return (((n+(n>>4))&0x0F0F0F0F)*0x01010101)>>24;}
inline float quick_sharpe_ratio(const float &adr, const int period, const float &std){return (adr/std)*128.0f;}//not for daily, need to adjust based on time period used
inline int Hash(int a,int b){return a*PRIME+b;}
inline Pii de_hash(int a){return MP(a%PRIME,a/PRIME);}

int read_in_data(const string name, float *O, float *H,float *L, float *C);
int get_state(const float *O, const float *H, const float *L, const float *C,const int idx, const int days_back,const int adj,const int adj2,float p0,float p1);
void cpu_fill_in(const float *O, const float *H, const float *L, const float *C,const int sz,int *StateArr,int *CPUfreq,float *ExpVal,
	const int days_back,const int days_forward,const int adj,const int adj2,float p0,float p1);
void cpu_scale(const int *CPUfreq,float *ExpVal,const int N);
Pii most_valuable_inputs(const float *expval, const int N);

//This kernel looks at each period and classifys the state with a bitmask for that period, looks forward to calc average price change associated with that pattern
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

int main(){
	char ch;
	const int N=PATTERN_SIZE;
	float *Open=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	float *High=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	float *Low=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	float *Close=(float *)malloc(ELEM_ESTIMATE*sizeof(float));
	int *Signal=(int *)malloc(ELEM_ESTIMATE*sizeof(int));
	int *PatternsCPU=(int *)malloc(ELEM_ESTIMATE*sizeof(int));
	int *PatternsGPU=(int *)malloc(ELEM_ESTIMATE*sizeof(int));
	int *CPUfreq=(int *)malloc(N*sizeof(int));
	int *GPUfreq=(int *)malloc(N*sizeof(int));
	float *CPUexpval=(float *)malloc(N*sizeof(float));
	float *GPUexpval=(float *)malloc(N*sizeof(float));
	string Pdata="Euro_2009-2012(5min).txt";
	const int num_elements=read_in_data(Pdata,Open,High,Low,Close);
	
	const int periods_back=9,lim=12,periods_forward=4;
	memset(PatternsCPU,-1,ELEM_ESTIMATE*sizeof(int));
	memset(PatternsGPU,-1,ELEM_ESTIMATE*sizeof(int));
	memset(CPUfreq,0,N*sizeof(int));
	memset(CPUexpval,0,N*sizeof(int));

	cout<<"Starting CPU testing:\n";
	UINT wTimerRes = 0;

	bool init = InitMMTimer(wTimerRes);
	DWORD startTime = timeGetTime(),GPUtime=0,CPUtime=0;

	cpu_fill_in(Open,High,Low,Close,num_elements,PatternsCPU,CPUfreq,CPUexpval,periods_back,periods_forward,5,6,1.0f,1.0f);
	cpu_scale(CPUfreq,CPUexpval,N);

	DWORD endTime = timeGetTime();
	CPUtime=endTime-startTime;
	cout<<"CPU timing fill: "<<CPUtime<< " ms.\n";
	DestroyMMTimer(wTimerRes, init);


	if(DO_TEST){
		cout<<"Starting GPU testing:\n";
		float *O,*H,*L,*C,*ExpVal;
		int *Patterns,*freq;
		const int num_bytes_data=ELEM_ESTIMATE*sizeof(float);
		//allocate data on GPU
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
		err=cudaMalloc((void **)&Patterns,ELEM_ESTIMATE*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&freq,N*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void **)&ExpVal,N*sizeof(float));
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
		err=cudaMemcpy(Patterns,PatternsGPU,ELEM_ESTIMATE*sizeof(int),_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		

		const int num_blocks=(num_elements+THREADS-1)/THREADS;

		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();

		fill_in_state<<<num_blocks,THREADS>>>(O,H,L,C,Patterns,freq,ExpVal,num_elements,periods_back,periods_forward,5,6,1.0f,1.0f);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		gpu_scale<<<((N+THREADS-1)/THREADS),THREADS>>>(freq,ExpVal,N);

		endTime = timeGetTime();
		GPUtime=endTime-startTime;
		cout<<"GPU timing fill: "<<GPUtime<<" ms.\n";
		DestroyMMTimer(wTimerRes, init);

		err=cudaMemcpy(PatternsGPU,Patterns,ELEM_ESTIMATE*sizeof(int),_DTH);
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

		//cout<<"\nGPU memory freed!\n";
	}
	//Error checking
	int bad=0;
	for(int i=periods_back;i<num_elements;i++){
		if(PatternsCPU[i]!=PatternsGPU[i])bad++;
	}

	int bad2=0,bad3=0;
	map<int,pair<float,float> > MapExp;//if there are errors, will show up here with index and the two bad values
	float best_long=0.0f,best_short=0.0f;
	int idx_long=-1,idx_short=-1,most_freq=0,idx_freq=-1;
	for(int i=0;i<N;i++){
		if(GPUfreq[i]!=CPUfreq[i])bad2++;
		if(fabs(GPUexpval[i]-CPUexpval[i])>0.0001f){
			MapExp[i]=make_pair(CPUexpval[i],GPUexpval[i]);
			++bad3;
		}
		if(GPUexpval[i]>best_long){
			best_long=GPUexpval[i];
			idx_long=i;
		}
		if(GPUexpval[i]<best_short){
			best_short=GPUexpval[i];
			idx_short=i;
		}
		if(GPUfreq[i]>most_freq){
			most_freq=GPUfreq[i];
			idx_freq=i;
		}
	}
	
	if(GPUtime==0){//sometime it takes 0 ms to fill, so cannot divide by zero and adjust here for simplicity
		CPUtime++;
		GPUtime++;
	}
	if(bad==0 && bad2==0 && bad3==0){
		cout<<"\nNo errors!\nCPU implementation == GPU CUDA implementation, and CUDA version was "<<float(CPUtime)/float(GPUtime)<< " times faster.\n\n";
		cout<<"Pattern #"<<idx_long<<" was associated with the greatest "<<periods_forward*5<<" minute INCREASE of "<<100.0f*best_long<<" %\n";
		cout<<"This Pattern occurred "<<GPUfreq[idx_long]<<" times.\n\n";
		cout<<"Pattern #"<<idx_short<<" was associated with the greatest "<<periods_forward*5<<" minute DECREASE of "<<100.0f*best_short<<" %\n";
		cout<<"This Pattern occurred "<<GPUfreq[idx_short]<<" times.\n\n";
		cout<<"Pattern #"<<idx_freq<<" was most frequent, with "<<most_freq<<" occurences, and an expected "<<periods_forward*5<<" min change of "<<100.0f*GPUexpval[idx_freq]<<" %\n\n";
		Pii get_best_inputs=most_valuable_inputs(GPUexpval,N);
		cout<<"Most valuable input predictor of short-term bullish moves was input #"<<get_best_inputs.first<<'\n';
		cout<<"Most valuable input predictor of short-term bearish moves was input #"<<get_best_inputs.second<<'\n';
	}else
		cout<<"\nError in calculation!\n";
	

	free(Open);
	free(High);
	free(Low);
	free(Close);
	free(Signal);
	free(PatternsCPU);
	free(PatternsGPU);
	free(CPUfreq);
	free(GPUfreq);
	free(CPUexpval);
	free(GPUexpval);

	MapExp.clear();
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
	}
	fclose(file);
	return num_elements;
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




