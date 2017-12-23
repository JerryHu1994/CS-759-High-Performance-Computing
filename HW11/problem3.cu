#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>

struct greater_than_five
{
	int threshold;
	greater_than_five(int t)	{threshold = t;}
	
	__host__ __device__
	bool operator()(int x)	{return x > threshold;}
};

//create the predicate functor (returns true for x > 10)
greater_than_five pred(5);


int main(int argc, char*argv[])
{
    thrust::host_vector<int> hostDay(15,0);    
    thrust::host_vector<int> hostSite(15,0);    
    thrust::host_vector<int> hostMeasure(15,0);    
    
    hostDay[0]= 0;  hostDay[1]= 0;  hostDay[2]= 1;
    hostDay[3]= 2;  hostDay[4]= 5;  hostDay[5]= 5;
    hostDay[6]= 6;  hostDay[7]= 6;  hostDay[ 8]= 7;
    hostDay[9]= 8;  hostDay[10]= 9; hostDay[11]= 9;
    hostDay[12]= 9; hostDay[13]=10; hostDay[14]=11;
    hostSite[0]= 2;  hostSite[1]= 3;  hostSite[2]= 0;
    hostSite[3]= 1;  hostSite[4]= 1;  hostSite[5]= 2;
    hostSite[6]= 0;  hostSite[7]= 1;  hostSite[8]= 2;
    hostSite[9]= 1;  hostSite[10]= 3;  hostSite[11]= 4;
    hostSite[12]= 0;  hostSite[13]= 1;  hostSite[14]= 2;
    hostMeasure[0]= 9;  hostMeasure[1]= 5;  hostMeasure[2]= 6;
    hostMeasure[3]= 3;  hostMeasure[4]= 3;  hostMeasure[5]= 8;
    hostMeasure[6]= 2;  hostMeasure[7]= 6;  hostMeasure[8]= 5;
    hostMeasure[9]=10;  hostMeasure[10]= 9;  hostMeasure[11]=11;
    hostMeasure[12]= 8;  hostMeasure[13]= 4;  hostMeasure[14]= 1;


    //copy the vector from host to device
    thrust::device_vector<int> deviceDay(15,0); 
    thrust::device_vector<int> deviceSite(15,0); 
    thrust::device_vector<int> deviceMeasure(15,0); 
    thrust::copy(hostDay.begin(),hostDay.end(),deviceDay.begin());
    thrust::copy(hostSite.begin(),hostSite.end(),deviceSite.begin());
    thrust::copy(hostMeasure.begin(),hostMeasure.end(),deviceMeasure.begin());
   
    using namespace thrust::placeholders;
    
    //problem a   
    //int count5 = thrust::count_if(deviceMeasure.begin(),deviceMeasure.end(),_1 > 5);
    int count5 = thrust::count_if(deviceMeasure.begin(),deviceMeasure.end(),pred);
    printf("%d\n",count5); 
    
    //problem b
    thrust::device_vector<int> sortSite(15,0);
    thrust::device_vector<int> measure(15,0);
    
    thrust::copy(deviceSite.begin(),deviceSite.end(),sortSite.begin());
    thrust::copy(deviceMeasure.begin(),deviceMeasure.end(),measure.begin());
    
    //sort the sites
    thrust::sort_by_key(sortSite.begin(),sortSite.end(),measure.begin());

    //reduce by key
    thrust::reduce_by_key(sortSite.begin(),sortSite.end(),measure.begin(),deviceSite.begin(),deviceMeasure.begin());

    thrust::copy(deviceSite.begin(),deviceSite.end(),hostSite.begin());
    thrust::copy(deviceMeasure.begin(),deviceMeasure.end(),hostMeasure.begin());
   
    for(int i=0;i<5;i++){
        if(i!=4){
			printf("%d ",hostMeasure[i]);
    	} else {
			printf("%d", hostMeasure[i]);
		}
	}
    printf("\n");

    return 0;

}
