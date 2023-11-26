
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//specify the value of K
#define K 7


//custom struct to store each image record in mnist dataset
//feutures are pointer to an array to  store features (They are 28X28 features for each image) of mnist dataset.
//out is class label for each image
typedef struct {
    int out;
    double *feutures;                   
} record_t;

//custom struct to store distance (distaance between perticular query image and reference image ) for each mnist Image.
//out is class label for the reference Image.
//distance feild is to store distance between perticular query image and reference Image.
typedef struct {
    int out;
    double distance;                   
} distance_t;

//distance_m is for manhattan_distance metric
double distance_m(record_t img1, record_t img2) {
    double dist = 0;
    //size of image in mnsit dataset is 28x28=784
    for (int i = 0; i < 784; i++) {
        double diff = img1.feutures[i] - img2.feutures[i];
        if(diff<0){
               diff=-1*diff;
        }
        dist += diff;
    }
    return dist;
}

//distance_e is for euclidean_distance metric
double distance_e(record_t img1, record_t img2) {
    double dist = 0;
    //size of image in mnsit dataset is 28x28=784
    for (int i = 0; i < 784; i++) {
        double diff = img1.feutures[i] - img2.feutures[i];
        dist += diff*diff;
    }
    return sqrt(dist);
}



//function to free up the record_t pointers after execution of knn
void free_images(record_t images[], int num_images) {
    for (int i = 0; i < num_images; i++) {
        free(images[i].feutures);
    }
}



//custom compare function used for qsort built in C
int compare(const void *a, const void *b)
{
    distance_t *da = (distance_t *)a;
    distance_t *db = (distance_t *)b;
    if (da->distance < db->distance) {
        return -1;
    } else if (da->distance > db->distance) {
        return 1;
    } else {
        return 0;
    }
}



//serial knn algorithm that takes query image, reference set which is named as dataset here , and the number of images in reference set
int knn(record_t test, record_t dataset[], int num_images) {
    //array to store distances between query point to each reference set and their class labels
    distance_t distances[num_images];
    
    for (int i = 0; i < num_images; i++) {
        //Euclidean distnaces
        distances[i].distance = distance_e(test, dataset[i]);
        distances[i].out = dataset[i].out;
    }
    //sorting distances using qsort 
    qsort(distances, num_images, sizeof(distance_t), compare);
    //array to store first k distances and there class labels.
    int arr[K];
    
    for(int ll=0;ll<K;ll++){
    arr[ll]=distances[ll].out;}
    
    int freq[10] = {0};
    int max_freq = 0, max_num = 0;
    for (int temp = 0; temp < K; temp++) {
        freq[arr[temp]]++;
        if (freq[arr[temp]] > max_freq) {
            max_freq = freq[arr[temp]];
            max_num = arr[temp];
        }
    }
    //max_num is prediction for the query(or test) image.
    return max_num;
}






int main(void) {


// Uncomment below to get the approximate time taken by the program including i/o 
// struct timespec start, end;
//     clock_gettime(CLOCK_MONOTONIC, &start);
    
    
    
    
    
    FILE *fp;
    // To store each row from the mnist train dataset(each row is image)
    record_t *row = malloc(60000 * sizeof(record_t));
    for(int k=0;k<60000;k++){
        row[k].feutures = malloc(784 * sizeof(double));
    }
    //To store each row from the mnist test dataset
        record_t *rowt = malloc(10000 * sizeof(record_t));
    for(int k=0;k<10000;k++){
        rowt[k].feutures = malloc(784 * sizeof(double));
    }
    
    size_t count = 0;
    fp = fopen("mnist_train.csv", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error reading file\n");
        return 1;
    }
    
int read = 0;
int records = 0;
int num_feutures=784;
do {
  read = fscanf(fp, "%d,", &row[records].out);

  for (int i = 0; i < num_feutures; i++) {
    read = fscanf(fp, "%lf,", &row[records].feutures[i]);
  }
    records++;
  if (ferror(fp)) {
    printf("Error reading file.\n");
   return 1;
  }  
} while (!feof(fp));     
        fclose(fp);
        printf("\n%d records read from mnist train set.\n\n", records-1);  

    fp = fopen("mnist_test.csv", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error reading file\n");
        return 1;
    }
    
read = 0;
records = 0;
num_feutures=784;
do {
  read = fscanf(fp, "%d,", &rowt[records].out);

  for (int i = 0; i < num_feutures; i++) {
    read = fscanf(fp, "%lf,", &rowt[records].feutures[i]);
  }
    records++;
  if (ferror(fp)) {
    printf("Error reading file.\n");
   return 1;
  }  
} while (!feof(fp));     
        fclose(fp);
        printf("\n%d records read from the mnist test set.\n\n", records-1); 
        
        
        
    int num_correct = 0;
    //10000 is length of test set
    for (int i = 0; i < (10000); i++) {
        //get the current query image
        record_t test_image = rowt[i];
        //calculate distance between query to all reference images in train set and predict the class label
        int predicted_label = knn(test_image, row, 60000);//60000 is length of train set
        if (predicted_label == test_image.out) {
            num_correct++;
        }
        printf("Query Image: %d,Predicted_label: %d,Actual Label: %d\n",i,predicted_label,test_image.out);
    }
    double accuracy = (double) num_correct / (10000);
    printf("Accuracy: %f\n", accuracy);
    //free the memory
    free_images(row, 60000);
    free_images(rowt, 10000);   
    
    
    
    // uncomment it to get the approximate time taken by the program including i/o 
    //     clock_gettime(CLOCK_MONOTONIC, &end);
    // double time_taken = (end.tv_sec - start.tv_sec) +
    //                     (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    //                      printf("Time taken: %f seconds\n", time_taken);
        return 0;
}              
