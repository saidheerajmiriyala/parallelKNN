// cuda streams used here is 4. 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<cuda.h>
#include <time.h>

#define K 1 // k for KNN 


#define TRAIN_SIZE (60000)
#define TEST_SIZE (10000)


typedef struct {
    int out;
    double *feutures;                   
} record_t;
//struct for storing  each datapoint in datasets, out is class , feutures is pointer to feuture array


typedef struct {
    int out;
    double distance;                   
} distance_t;
//struct for storing distance value and corresponding class label after findind distance between two points

typedef struct {
    distance_t *dist_s0; 
    distance_t *dist_s1;
    distance_t *dist_s2;
    distance_t *dist_s3;
} final_t;
//struct of type final_t contains 4 arrays of distances from each 4 streams

typedef struct{
    distance_t *arr_s0;
    distance_t *arr_s1;
    distance_t *arr_s2;
    distance_t *arr_s3;
}tempstruct;
//tempporary struct to trim the distances arrays to only first K smallest elements for each testpoint


typedef struct{
    distance_t *arr;
    int *arrfinal;
    int *freq;
}tempstruct2;
//temporary struct used by sorting kernel for final kernel for each test point






__device__ void selection_sort(distance_t arr[], int n) {
    int i, j, min_idx;
    for (i = 0; i < n-1; i++) {
        min_idx = i;
        for (j = i+1; j < n; j++) {
            if (arr[j].distance < arr[min_idx].distance) {
                min_idx = j;
            }
        }
        // Swap elements
        distance_t temp = arr[i];
        arr[i] = arr[min_idx];
        arr[min_idx] = temp;

        // Check if Kth element has been sorted
        if (i == K-1) {
            return;
        }
    }
}
//selection sort limited to first k iterations leading to first k smallest elements













//euclidean distance
__device__ double distanceBetweenImages_e(double *pixels1, double *pixels2) {
    double dist = 0;
    for (int i = 0; i < 784; i++) {
        double diff = pixels1[i] - pixels2[i];
        dist += diff*diff;
    }
    return sqrt(dist);
}


//manhatan distance
__device__ double distanceBetweenImages_m(double *pixels1, double *pixels2) {
    double dist = 0;
    for (int i = 0; i < 784; i++) {
        double diff = pixels1[i] - pixels2[i];
        if(diff<0){
        diff=-1*diff;}
        dist += diff;
    }
    return dist;
}



//to free the  device allocated arrays
void free_arrays(record_t images[], int num_images) {
    for (int i = 0; i < num_images; i++) {
        cudaFree(images[i].feutures);
    }
}




//distance kernel that uses grid stride loop 
__global__ void distance_kernel(record_t *train_data, record_t test_point, distance_t *distancesss,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < size; i+=stride) {  distancesss[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point.feutures);
    distancesss[i].out=train_data[i].out;
    
}

}


//sorting kernel
__global__ void sorting_kernel(tempstruct2 *f4karrays,tempstruct *allarrays,final_t *final_dist_all, int N, int *results,int t_s0){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < N; mm+=stride){
      //each thread sorts using k limit selection sort
      //each thread sorts all distance arrays sequencially
    selection_sort(final_dist_all[mm].dist_s0, t_s0);
    selection_sort(final_dist_all[mm].dist_s1, t_s0);
    selection_sort(final_dist_all[mm].dist_s2, t_s0);
    selection_sort(final_dist_all[mm].dist_s3, t_s0);
    //takes the first k elements from each distance arrays and combine to get array of 4K
    for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s0[ll]=final_dist_all[mm].dist_s0[ll];}
        for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s1[ll]=final_dist_all[mm].dist_s1[ll];}
        for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s2[ll]=final_dist_all[mm].dist_s2[ll];}
        for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s3[ll]=final_dist_all[mm].dist_s3[ll];}
    int i, j = 0;
    for (i = 0; i < K; i++) {
        f4karrays[mm].arr[j++] = allarrays[mm].arr_s0[i];
    }
    for (i = 0; i < K; i++) {
        f4karrays[mm].arr[j++] = allarrays[mm].arr_s1[i];
    }
    for (i = 0; i < K; i++) {
        f4karrays[mm].arr[j++] = allarrays[mm].arr_s2[i];
    }
    for (i = 0; i < K; i++) {
        f4karrays[mm].arr[j++] = allarrays[mm].arr_s3[i];
    }  
    selection_sort(f4karrays[mm].arr, 4*K);
    for(int ll=0;ll<K;ll++){
    f4karrays[mm].arrfinal[ll]=f4karrays[mm].arr[ll].out;}
    // int freq[10] = {0};
    int max_freq = 0, max_num = 0;
    for (int i = 0; i < K; i++) {
        f4karrays[mm].freq[f4karrays[mm].arrfinal[i]]++;
        if (f4karrays[mm].freq[f4karrays[mm].arrfinal[i]] > max_freq) {
            max_freq = f4karrays[mm].freq[f4karrays[mm].arrfinal[i]];
            max_num = f4karrays[mm].arrfinal[i];
        }
    }

    results[mm]=max_num;}








}

int main() {
  //row is train set
  //rowt is test dataset
    FILE *fp;
    int n = 60000;
    
        record_t *row;
    cudaMallocManaged(&row, n * sizeof(record_t));
        for (int k = 0; k < n; k++) {
    cudaMallocManaged(&(row[k].feutures), 784 * sizeof(double));
}
   



    fp = fopen("mnist_train(1).csv", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error reading file\\\\n");
        return 1;
    }

    int records = 0;
    int read=0;
  
do {
  read = fscanf(fp, "%d,", &row[records].out);

  for (int i = 0; i < 784; i++) {
    read = fscanf(fp, "%lf,", &row[records].feutures[i]);
  }

 
    records++;
  

  if (ferror(fp)) {
    printf("Error reading file.\\\\n");
   return 1;
  }  
} while (!feof(fp)); 
        fclose(fp);
        
        printf("\\\\n%d records read.\\\\n\\\\n", records);  



     

             record_t *rowt;
    cudaMallocManaged(&rowt, 10000 * sizeof(record_t));
        for (int k = 0; k < 10000; k++) {
    cudaMallocManaged(&(rowt[k].feutures), 784 * sizeof(double));
}
        
           fp = fopen("mnist_test.csv", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error reading file\\\\n");
        return 1;
    }

     records = 0;
     read=0;
  
do {
  read = fscanf(fp, "%d,", &rowt[records].out);
  for (int i = 0; i < 784; i++) {
    read = fscanf(fp, "%lf,", &rowt[records].feutures[i]);
  }

 
    records++;
  

  if (ferror(fp)) {
    printf("Error reading file.\\\\n");
   return 1;
  }  
} while (!feof(fp)); 
        fclose(fp);
        
        printf("\\\\n%d records read.\\\\n\\\\n", records);
       
       
       struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);

        
           

 //streams
cudaStream_t s1;
cudaStream_t s2;
cudaStream_t s3;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
cudaStreamCreate(&s3);

int n1=60000;
        final_t *final_dist_all;
    cudaMallocManaged(&final_dist_all, TEST_SIZE * sizeof(final_t));
        for (int k = 0; k < TEST_SIZE; k++) {
    cudaMallocManaged(&(final_dist_all[k].dist_s0), (n1/4) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s1), (n1/4) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s2), (n1/4) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s3), (n1/4) * sizeof(distance_t));
}





int t_s1=(n1/4);
int t_s2=(n1/4);
int t_s3=(n1/4);
int t_s0=(n1/4);
// record_t *part1_s1;
// record_t *part1_s2;
// record_t *part1_s3;
// record_t *part1_s0;
// cudaMallocManaged(&part1_s1, t_s1 * sizeof(record_t));
// cudaMallocManaged(&part1_s2, t_s2 * sizeof(record_t));
// cudaMallocManaged(&part1_s3, t_s3 * sizeof(record_t));
// cudaMallocManaged(&part1_s0, t_s0 * sizeof(record_t));



// cudaMemcpy(part1_s0, row, t_s1 * sizeof(record_t), cudaMemcpyDefault);
// cudaMemcpy(part1_s1, row + t_s1, t_s1 * sizeof(record_t), cudaMemcpyDefault);
// cudaMemcpy(part1_s2, row + 2 * t_s1, t_s1 * sizeof(record_t), cudaMemcpyDefault);
// cudaMemcpy(part1_s3, row + 3 * t_s1, t_s1 * sizeof(record_t), cudaMemcpyDefault);
    


int no_correct=0;
for(int mm=0;mm<(10000);mm++){
    distance_kernel<<<20,64>>>(row, rowt[mm],final_dist_all[mm].dist_s0,t_s0);
    distance_kernel<<<20,64,0,s1>>>(row+t_s1, rowt[mm],final_dist_all[mm].dist_s1,t_s1);
    distance_kernel<<<20,64,0,s2>>>(row+2*t_s1, rowt[mm],final_dist_all[mm].dist_s2,t_s1);
    distance_kernel<<<20,64,0,s3>>>(row+3*t_s1, rowt[mm],final_dist_all[mm].dist_s3,t_s1);
        }
cudaDeviceSynchronize();






int *results;
cudaMallocManaged(&results, (10000) * sizeof(int));
        tempstruct *allarrays;
    cudaMallocManaged(&allarrays, TEST_SIZE * sizeof(tempstruct));
        for (int k = 0; k < TEST_SIZE; k++) {
    cudaMallocManaged(&(allarrays[k].arr_s0), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s1), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s2), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s3), (K) * sizeof(distance_t));
}



        tempstruct2 *f4karrays;
    cudaMallocManaged(&f4karrays, TEST_SIZE * sizeof(tempstruct2));
        for (int k = 0; k < TEST_SIZE; k++) {
    cudaMallocManaged(&(f4karrays[k].arr), (4*K) * sizeof(distance_t));
    cudaMallocManaged(&(f4karrays[k].arrfinal), (K) * sizeof(int));
    cudaMallocManaged(&(f4karrays[k].freq), (10) * sizeof(int));
}

sorting_kernel<<<20,256>>>(f4karrays,allarrays,final_dist_all,(10000),results, t_s0);

cudaDeviceSynchronize();
for(int mmm=0;mmm<(10000);mmm++){
    
    if(results[mmm]==rowt[mmm].out){
        no_correct++;
    }
}

            double accuracy = (double) no_correct / (TEST_SIZE);
    printf("Accuracy: %f\\\\n", accuracy);
                clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1000000000.0;

                         printf("Time taken: %f seconds\\\\n", time_taken); 
cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
cudaStreamDestroy(s3);


    free_arrays(row,60000);
    free_arrays(rowt,10000);
    cudaFree(row);
    cudaFree(rowt);
    for (int k = 0; k < TEST_SIZE; k++) {
    cudaFree(final_dist_all[k].dist_s0);
    cudaFree(final_dist_all[k].dist_s1);
    cudaFree(final_dist_all[k].dist_s2);
    cudaFree(final_dist_all[k].dist_s3);
}

cudaFree(final_dist_all);

for (int k = 0; k < TEST_SIZE; k++) {
    cudaFree(f4karrays[k].arr);
    cudaFree(f4karrays[k].arrfinal);
    cudaFree(f4karrays[k].freq);
}
cudaFree(f4karrays);


for (int k = 0; k < TEST_SIZE; k++) {
    cudaFree(allarrays[k].arr_s0);
    cudaFree(allarrays[k].arr_s1);
    cudaFree(allarrays[k].arr_s2);
    cudaFree(allarrays[k].arr_s3);
}
cudaFree(allarrays);
cudaFree(results);
}
