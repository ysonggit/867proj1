/**************************
  Yang Song
  song24@email.sc.edu
  CSCE 867 Project 1
**************************/
#include "calib.h"

int main(){
    
  string file2d ("2Dpoints.txt");
  myMatrix M2d = inputMat(file2d); // 72 x 2
 
  string file3d1 ("3Dpoints_part1.txt");
  myMatrix M3d1 = inputMat(file3d1); // 72 x 3
  int Num = int(M3d1.size());
  int N_calib = 60;
  int N_error = Num - N_calib;
  
  /////////////////////////////// Part I ////////////////////////////////////////
  Mat P = findProjectionMat(M2d, M3d1, N_calib);
 
  findParameters(P);
  float epsilon = findProjectionError(M2d, M3d1, P, N_error);
  

  /////////////////////////////// Part II ///////////////////////////////////////
  string file3d2 ("3Dpoints_part2.txt");
  myMatrix M3d2 = inputMat(file3d2);

  // Step 1:  partiation all points into a traing set and testing set
  int K = 10;
     
  ransac(M2d, M3d2, K, Num);
  
  return 0;

}
