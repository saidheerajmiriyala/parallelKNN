#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<cuda.h>
#include <time.h>

#define K 7
#define  noofdimensions 784
#define TRAIN_SIZE (60000)
#define TEST_SIZE (10000)



typedef struct {
    int out;
    double *feutures;                   
} record_t;



typedef struct {
    int out;
    double distance;                   
} distance_t;


typedef struct {
    distance_t *dist_s0; 
    distance_t *dist_s1;
    distance_t *dist_s2;
    distance_t *dist_s3;
    distance_t *dist_s4;
    distance_t *dist_s5;
} final_t;


typedef struct{
    distance_t *arr_s0;
    distance_t *arr_s1;
    distance_t *arr_s2;
    distance_t *arr_s3;
    distance_t *arr_s4;
    distance_t *arr_s5;

}tempstruct;



typedef struct{
    distance_t *arr;
    int *arrfinal;
    int *freq;
}tempstruct2;


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

__device__ double distanceBetweenImages_e(double *pixels1, double *pixels2) {
    double dist = 0;
    for (int i = 0; i < noofdimensions; i++) {
        double diff = pixels1[i] - pixels2[i];
        dist += diff*diff;
    }
    return sqrt(dist);
}



__device__ double distanceBetweenImages_m(double *pixels1, double *pixels2) {
    double dist = 0;
    for (int i = 0; i < noofdimensions; i++) {
        double diff = pixels1[i] - pixels2[i];
        if(diff<0){
        diff=-1*diff;}
        dist += diff;
    }
    return dist;
}



void free_images(record_t images[], int num_images) {
    for (int i = 0; i < num_images; i++) {
        cudaFree(images[i].feutures);
    }
}



__global__ void knn_slaves0(record_t *train_data, record_t *test_point, final_t *final_dist_all,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < TEST_SIZE; mm+=stride) { 
    for(int i =0;i<size;i++){
    final_dist_all[mm].dist_s0[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point[mm].feutures);
    final_dist_all[mm].dist_s0[i].out=train_data[i].out;
    }
    
}

}


__global__ void knn_slaves1(record_t *train_data, record_t *test_point, final_t *final_dist_all,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < TEST_SIZE; mm+=stride) { 
    for(int i =0;i<size;i++){
    final_dist_all[mm].dist_s1[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point[mm].feutures);
    final_dist_all[mm].dist_s1[i].out=train_data[i].out;
    }
    
}

}

__global__ void knn_slaves2(record_t *train_data, record_t *test_point, final_t *final_dist_all,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < TEST_SIZE; mm+=stride) { 
    for(int i =0;i<size;i++){
    final_dist_all[mm].dist_s2[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point[mm].feutures);
    final_dist_all[mm].dist_s2[i].out=train_data[i].out;
    }
    
}

}

__global__ void knn_slaves3(record_t *train_data, record_t *test_point, final_t *final_dist_all,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < TEST_SIZE; mm+=stride) { 
    for(int i =0;i<size;i++){
    final_dist_all[mm].dist_s3[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point[mm].feutures);
    final_dist_all[mm].dist_s3[i].out=train_data[i].out;
    }
    
}

}



__global__ void knn_slaves4(record_t *train_data, record_t *test_point, final_t *final_dist_all,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < TEST_SIZE; mm+=stride) { 
    for(int i =0;i<size;i++){
    final_dist_all[mm].dist_s4[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point[mm].feutures);
    final_dist_all[mm].dist_s4[i].out=train_data[i].out;
    }
    
}

}


__global__ void knn_slaves5(record_t *train_data, record_t *test_point, final_t *final_dist_all,int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < TEST_SIZE; mm+=stride) { 
    for(int i =0;i<size;i++){
    final_dist_all[mm].dist_s5[i].distance=distanceBetweenImages_e(train_data[i].feutures,test_point[mm].feutures);
    final_dist_all[mm].dist_s5[i].out=train_data[i].out;
    }
    
}

}


__global__ void finalkernel(tempstruct2 *f4karrays,tempstruct *allarrays,final_t *final_dist_all, int N, int *results,int t_s0){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int mm = tid; mm < N; mm+=stride){
    selection_sort(final_dist_all[mm].dist_s0, t_s0);
    selection_sort(final_dist_all[mm].dist_s1, t_s0);
    selection_sort(final_dist_all[mm].dist_s2, t_s0);
    selection_sort(final_dist_all[mm].dist_s3, t_s0);
    selection_sort(final_dist_all[mm].dist_s4, t_s0);
    selection_sort(final_dist_all[mm].dist_s5, t_s0);
    for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s0[ll]=final_dist_all[mm].dist_s0[ll];}
        for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s1[ll]=final_dist_all[mm].dist_s1[ll];}
        for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s2[ll]=final_dist_all[mm].dist_s2[ll];}
        for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s3[ll]=final_dist_all[mm].dist_s3[ll];}
    for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s4[ll]=final_dist_all[mm].dist_s4[ll];}
    for(int ll=0;ll<K;ll++){
    allarrays[mm].arr_s5[ll]=final_dist_all[mm].dist_s5[ll];}
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
    for (i = 0; i < K; i++) {
        f4karrays[mm].arr[j++] = allarrays[mm].arr_s4[i];
    } 
    for (i = 0; i < K; i++) {
        f4karrays[mm].arr[j++] = allarrays[mm].arr_s5[i];
    }
    selection_sort(f4karrays[mm].arr, 6*K);
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
        
        printf("\\\\n%d records read.\\\\n\\\\n", records-1);  




n2=10000;
        record_t *rowt;
    cudaMallocManaged(&rowt, n2 * sizeof(record_t));
        for (int k = 0; k < n2; k++) {
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
        
        printf("\\\\n%d records read.\\\\n\\\\n", records - 1);  


n1=60000;
cudaStream_t s1;
cudaStream_t s2;
cudaStream_t s3;
cudaStream_t s4;
cudaStream_t s5;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
cudaStreamCreate(&s3);
cudaStreamCreate(&s4);
cudaStreamCreate(&s5);

        final_t *final_dist_all;
    cudaMallocManaged(&final_dist_all, TEST_SIZE * sizeof(final_t));
        for (int k = 0; k < TEST_SIZE; k++) {
    cudaMallocManaged(&(final_dist_all[k].dist_s0), (n1/6) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s1), (n1/6) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s2), (n1/6) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s3), (n1/6) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s4), (n1/6) * sizeof(distance_t));
    cudaMallocManaged(&(final_dist_all[k].dist_s5), (n1/6) * sizeof(distance_t));
}





int t_s1=(n1/6);
int t_s2=(n1/6);
int t_s3=(n1/6);
int t_s0=(n1/6);


int no_correct=0;


struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);

for(int i=0; i<10000;i++){

    knn_slaves1<<<20,64,0,s1>>>(row, rowt[i],final_dist_all,t_s1);
    knn_slaves2<<<20,64,0,s2>>>(row+(n1/6), rowt[i],final_dist_all,t_s1);
    knn_slaves3<<<20,64,0,s3>>>(row+2*(n1/6), rowt[i],final_dist_all,t_s1);
    knn_slaves4<<<20,64,0,s4>>>(row+2*(n1/6), rowt[i],final_dist_all,t_s1);
    knn_slaves5<<<20,64,0,s5>>>(row+4*(n1/6), rowt[i],final_dist_all,t_s1);
    knn_slaves0<<<20,64>>>(row+5*(n1/6), rowt[i],final_dist_all,t_s1);
    
    }
        











int *results;
cudaMallocManaged(&results, (noofelements-n1) * sizeof(int));
        tempstruct *allarrays;
    cudaMallocManaged(&allarrays, TEST_SIZE * sizeof(tempstruct));
        for (int k = 0; k < TEST_SIZE; k++) {
    cudaMallocManaged(&(allarrays[k].arr_s0), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s1), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s2), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s3), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s4), (K) * sizeof(distance_t));
    cudaMallocManaged(&(allarrays[k].arr_s5), (K) * sizeof(distance_t));
}

        tempstruct2 *f4karrays;
    cudaMallocManaged(&f4karrays, TEST_SIZE * sizeof(tempstruct2));
        for (int k = 0; k < TEST_SIZE; k++) {
    cudaMallocManaged(&(f4karrays[k].arr), (6*K) * sizeof(distance_t));
    cudaMallocManaged(&(f4karrays[k].arrfinal), (K) * sizeof(int));
    cudaMallocManaged(&(f4karrays[k].freq), (10) * sizeof(int));
}
cudaDeviceSynchronize();
finalkernel<<<45,256>>>(f4karrays,allarrays,final_dist_all,(noofelements-n1),results, t_s0);

cudaDeviceSynchronize();
for(int mmm=0;mmm<(noofelements-n1);mmm++){
    printf("%d,%d\n",results[mmm],part2[mmm].out);
    if(results[mmm]==part2[mmm].out){
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
cudaStreamDestroy(s4);
cudaStreamDestroy(s5);
    free_images(row,60000);
    cudaFree(row);
    free_images(rowt,10000);
    cudaFree(rowt);
    
    for (int k = 0; k < TEST_SIZE; k++) {
    cudaFree(final_dist_all[k].dist_s0);
    cudaFree(final_dist_all[k].dist_s1);
    cudaFree(final_dist_all[k].dist_s2);
    cudaFree(final_dist_all[k].dist_s3);
    cudaFree(final_dist_all[k].dist_s4);
    cudaFree(final_dist_all[k].dist_s5);
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
    cudaFree(allarrays[k].arr_s4);
    cudaFree(allarrays[k].arr_s5);
}
cudaFree(allarrays);
cudaFree(results);



    return 0;

}
