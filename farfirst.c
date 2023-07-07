#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "mat.h"
#include "vec.h"


int main (int argc, char** argv) {
       if (argc < 2) {
	printf ("Command usage : %s %s\n",argv[0],"k");
	return 1;
    }

    int k = atoi(argv[1]);

    int columns; 
    int rows; 

   scanf("%d%d",&rows,&columns);

  
   struct mat_s PMatrix;
   mat_calloc(&PMatrix, rows, columns);

   mat_read(&PMatrix);
    

   

     struct mat_s CMatrix;


    mat_calloc(&CMatrix, k, columns);

    for(int i=0;i<columns;i++){
        CMatrix.data[i] = PMatrix.data[i];

     }
     



    
 double min_cost = DBL_MAX; 

    for(int i =1;i<k;i++){
             double maxCost = 0;


        for(int j=0;j<rows;j++){
              struct vec_s currentVector;
            vec_calloc(&currentVector, columns);
            mat_get_row(&PMatrix, &currentVector, j);
            double cost=DBL_MAX;

            for(int u=0;u<i;u++){
            
                struct vec_s compareVector;
                vec_calloc(&compareVector, columns);
                 mat_get_row(&CMatrix, &compareVector, u);
                 double currentDistance = vec_dist_sq(&currentVector, &compareVector);
                 if(currentDistance<cost){
                    cost = currentDistance;
                 }
            }


            if(maxCost<cost){
                maxCost= cost;

                if(cost<min_cost){
                    min_cost=cost;
                }

                for(int index=0;index<columns;index++){
                    CMatrix.data[i+index] = PMatrix.data[(j*columns)+index];

                }
            }
            

        }




  

    }


  


printf("# approximate optimal cost = %lf\n", min_cost);
 printf("# approx optimal centers :\n");
 mat_print(&CMatrix);






    return 0;
}