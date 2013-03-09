
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <set>
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

typedef vector<vector<float> > myMatrix;

/* fileName should be of string type */
myMatrix inputMat(string & fileName){
    ifstream infile;
    infile.open(fileName.c_str());
    myMatrix a;
    istringstream istr;
    string str;
    vector<float> tempvec;
    while(getline(infile, str)){
        istr.str(str);
        float tmp;
        while(istr>>tmp){
            tempvec.push_back(tmp);
        }
        a.push_back(tempvec);
        tempvec.clear();
        istr.clear();
    }
    infile.close();
    return a;
}


void showMat(myMatrix M){
    for(int i=0; i<int(M.size()); i++){
        for(int j=0; j<int(M.at(i).size()); j++){
            cout<<M.at(i).at(j)<<" "<<flush;
        }
        cout<<endl;
    }
}

void displayMat(Mat M){
    cout<<"matrix size [ "<<M.rows<<" , "<<M.cols<<" ]"<<endl;
    for(int i=0; i<M.rows; i++){
        for(int j=0; j< M.cols; j++){
            cout<<M.at<float>(i,j)<<" ";
        }
        cout<<endl;
    }
    cout<<"***********************************************"<<endl;
}

Mat matFromMatrix(myMatrix M_in, int offset, int M, int N, int flag){
    // flag = 1 means Homogenous Coordinates, otherwise, flag = 0
    int col = N+flag;
    Mat M_out = Mat(M, col, CV_32F);
    for(int i=0; i<M_out.rows; i++){
        for(int j=0; j<N; j++){
            M_out.at<float>(i, j) = M_in.at(i+offset).at(j);
        }
        if(flag==1){
            M_out.at<float>(i, N) = 1;
        }
    }
    return M_out;
}

////////////////////////////////////////////////////////////////////
//
//                Functions for Calibration
//
///////////////////////////////////////////////////////////////////
/* Use two input matrices: one 2d image points matrix, one 3d object
   points matrix */
Mat findProjectionMat(myMatrix M2d, myMatrix M3d1, int N_calib){
    // use only 60 points for calibration: N_calib = 60
    // so A is a 120 x 12 matrix
  
    myMatrix A;
    for(int i=0; i<N_calib; i++){
        // get one 2d point each iteration
        float u_i = M2d.at(i).at(0);
        float v_i = M2d.at(i).at(1);
        // get one 3d point each time
        vector<float> M_w = M3d1.at(i);
   
        vector<float> evenrow;
        // push back 12 numbers in a line one-by-one 
        for(int j=0; j<3; j++){
            evenrow.push_back(M_w.at(j));
        }
        evenrow.push_back(1);
        for(int j=0; j<4; j++){
            evenrow.push_back(0);
        }
        for(int j=0; j<3; j++){
            evenrow.push_back(-M_w.at(j)*u_i);
        }
        evenrow.push_back(-u_i);
        A.push_back(evenrow);
    
        vector<float> oddrow;
        for(int k=0; k<4; k++){
            oddrow.push_back(0);
        }
      
        for(int k=0; k<3; k++){
            oddrow.push_back(M_w.at(k));
        }
        oddrow.push_back(1);
        for(int k=0; k<3; k++){
            oddrow.push_back(-M_w.at(k)*v_i);
        }
        oddrow.push_back(-v_i);
        A.push_back(oddrow);
    }
    //showMat(A);
    /* Can NOT directly use my own vector as InputArray,
       so I do this brute conversion    */
    Mat Amat = matFromMatrix(A, 0, (2*N_calib), 12, 0);
    //displayMat(Amat);
  
    Mat W, U, Vt;
    SVD::compute(Amat, W, U, Vt);
    bool dflag = false;
    if(dflag){
        cout<<"A ["<<Amat.rows<<" , "<<Amat.cols<<"]"<<endl;
        displayMat(Amat);
        cout<<"-------------------------"<<endl;
        cout<<"singular values W ["<<W.rows<<" , "<<W.cols<<"]"<<endl;
        displayMat(W);
        cout<<"-------------------------"<<endl;
        cout<<"LEFT singular vector U ["<<U.rows<<" , "<<U.cols<<"]"<<endl;
        displayMat(U);
        cout<<"-------------------------"<<endl;
        cout<<"RIGHT singular vector Vt ["<<Vt.rows<<" , "<<Vt.cols<<"]"<<endl;
        displayMat(Vt);
        cout<<"-------------------------"<<endl;
        // display last row of Vt
        cout<<"last column of V :"<<endl;
        for(int j=0; j< Vt.cols; j++){
            cout<<Vt.at<float>(Vt.rows-1, j)<<endl;
        }
        // use 12 points for error checking
    }
    Mat vv = Mat(1, 12, CV_32F);
    float alpha =1/ sqrt(pow(Vt.at<float>(Vt.rows-1,8),2)+pow(Vt.at<float>(Vt.rows-1,9),2)+pow(Vt.at<float>(Vt.rows-1,10), 2));
    //  cout<<"alpha is "<<alpha<<endl;
    vv.row(0) = Vt.row(11)*alpha;

    Mat P = Mat(3, 4, CV_32F);
    for (int k=0; k<3; k++){
        for(int i=0; i<4; i++){
            P.at<float>(k,i) = vv.at<float>(i+k*4);
        }
    }
    //cout<<endl<<pow(vv.at<float>(8),2)+pow(vv.at<float>(9),2)+pow(vv.at<float>(10), 2); 

    return P;
}


// input must be projection matrix p
void findParameters(Mat P){
   
    Mat p3 = Mat(3, 1, CV_32F);
    Mat p1 = Mat(3, 1, CV_32F);
    Mat p2 = Mat(3, 1, CV_32F);
    for(int i=0; i<3; i++){
        p1.at<float>(i) = P.at<float>(0,i);
        p2.at<float>(i) = P.at<float>(1,i);
        p3.at<float>(i) = P.at<float>(2,i);
    }
    // sign is +1, so r3 = p3
    Mat r3 = p3;
    Mat p1p3 =  p1.t() * p3;
    float u_0 = p1p3.at<float>(0);
    Mat p2p3 = p2.t() * p3;
    float v_0 = p2p3.at<float>(0);

    float t_z = P.at<float>(2,3);
    float p14 = P.at<float>(0,3);
    float p24 = P.at<float>(1,3);

    Mat p1p1 = p1.t() * p1;
    Mat p2p2 = p2.t() * p2;
    float f_u = sqrt(pow(p1p1.at<float>(0),2) - pow(u_0, 2));
    float f_v = sqrt(pow(p2p2.at<float>(0),2) - pow(v_0, 2));
        
    float t_x = -(p14 - u_0*t_z)/f_u;
    float t_y = -(p24 - v_0*t_z)/f_v;
    Mat T = Mat(3,1, CV_32F);
    T.at<float>(0) = t_x;
    T.at<float>(1) = t_y;
    T.at<float>(2) = t_z;
    Mat r1 = -(p1 - u_0 * p3)/f_u;
    Mat r2 = -(p2 - v_0 * p3)/f_v;
    Mat R  = Mat(3,3, CV_32F);
    R.row(0) = r1.t();
    R.row(1) = r2.t();
    R.row(2) = r3.t();
    
    cout<<"--------------------------------------"<<endl;
    cout<<"Projection Matrix P "<<endl;
    cout<<"--------------------------------------"<<endl;
    displayMat(P);
    cout<<"--------------------------------------"<<endl;
    cout<<"Rotation Matrix R"<<endl;
    cout<<"--------------------------------------"<<endl;
    displayMat(R);
    cout<<"--------------------------------------"<<endl;
    cout<<"Translation Parameter Vector T"<<endl;
    displayMat(T);
    cout<<"--------------------------------------"<<endl;
    cout<<"Intrinsic Parameters "<<endl;
    cout<<"--------------------------------------"<<endl;
    cout<<" f_u = "<<f_u<<endl;
    cout<<" f_v = "<<f_v<<endl;
    cout<<" u_0 = "<<u_0<<endl;
    cout<<" v_0 = "<<v_0<<endl;
    cout<<"--------------------------------------"<<endl;
}


/* M1 M2 must be the same size */
float computeError(Mat M1, Mat M2){
    // ignore size check here
    Mat E = M1 - M2;
    int N = E.rows; // number of points

    float sum = 0;
    for(int i = 0; i<E.rows; i++){
        for(int j=0; j<E.cols; j++){
            
            sum += pow(E.at<float>(i,j), 2);
        }
    }
    float epsilon = sum / N;
    return epsilon;
}


float  findProjectionError(myMatrix M2d, myMatrix M3d1, Mat P, int N){
    /* Projection Error */
    /* Use last N points for error checking. Let Mg the matrix of points from
       ground truth */
    
    int offset = int(M3d1.size()) - N;
       
    Mat Mg = matFromMatrix(M3d1, offset, N, 3, 1);
    /* Let Me be the matrix of points estimated by projection matrix P */
    Mat Me = P * Mg.t();
    /* Me = lamda x [u,v,1]' 
       lamda = r3_t x [x, y, z]' + t_z
    */
    Mat M_e = Me.t();
 
    Mat Mi_hat = Mat(N, 2, CV_32F);
    for(int i=0; i<N; i++){
        float lamda = M_e.at<float>(i, 2);
        for(int j=0; j<2; j++){
            Mi_hat.at<float>(i, j) = M_e.at<float>(i,j)/lamda;          
        }
    }
  
    /* Let Mi be the matrix of image points */
    Mat Mi = matFromMatrix(M2d, offset, N, 2, 0);

    float epsilon = computeError(Mi, Mi_hat);
    cout<<"\nProjection error  \n"<< epsilon <<endl;
    return epsilon;
}


////////////////////////////////////////////////////////////////////
//
//                     Functions for RANSAC 
//
///////////////////////////////////////////////////////////////////
void showSet(set<int> s){
    set<int>::iterator iter;
    for(iter = s.begin(); iter!=s.end(); ++iter){
        cout<<" "<<*iter;
    }
    cout<<endl;
}

// randomly choose m non-repeated number from 1 to n
set<int> train_idx(int m, int n){
    set<int> s;
    while(1){
        int r = rand()% n;
        s.insert(r);
        if(s.size() == m ){
            break;
        }
    }
    bool dflag = false;
    if(dflag){
        cout<<"randomly choose "<< m <<" points into training set"<<endl;
        showSet(s);
    }
    return s;
}

// input t is a set, training set indices
// input Num is the number of total points
set<int> test_idx(set<int> t, int Num){
    set<int> s;
    for(int i=0; i<Num; i++){
        s.insert(i);
    }

    set<int>::iterator it;
    vector<int> v_t;
    // re-store value of set t to vector v_t
    for(it=t.begin(); it!=t.end(); ++it){
        v_t.push_back(*it);
    }
    
    for(int i=0; i<int(v_t.size()); i++){
        s.erase(s.find(v_t.at(i)));
        //        cout<<" "<<v_t.at(i);
    }

    bool dflag = false;
    if(dflag){
        showSet(s);
    }
    return s;
}

myMatrix randMatFromMatrix(myMatrix M_in, set<int> s){
    set<int>::iterator it; //*it is the row number from M_in
    int m = int(s.size());
    int n = int(M_in.at(0).size());
    myMatrix M_out;
    it = s.begin();
    for(int i= 0; i< m; i++){
        vector<float> v_in;
        for(int j=0; j<n; j++){
            v_in.push_back( M_in.at(*it).at(j));
        }
        ++it;
        M_out.push_back(v_in);
    }
    bool dflag = false;
    if(dflag){
        showMat(M_out);
    }
    return M_out;
}

/* M2d is image point matrix for testing set, 
   M3d2 is the object point matrix for testing set
   P is the projection matrix  computed from training set
*/
vector<float> pointProjectionError(myMatrix M2d, myMatrix M3d2, Mat P){
    //compute vector v from P
    Mat v = Mat(12, 1, CV_32F);
    for(int i=0; i<3; i++){
        v.at<float>(i) = P.at<float>(0, i);
        v.at<float>(i+4) = P.at<float>(1, i);
        v.at<float>(i+8) = P.at<float>(2, i);
    }
    v.at<float>(3) = P.at<float>(0, 3);
    v.at<float>(7) = P.at<float>(1, 3);
    v.at<float>(11) = P.at<float>(2, 3);
    float lamda = P.at<float>(2,3);
    //displayMat(v);

    int N = int(M3d2.size()); // number of points in testing set
    vector<float> epsilon;
    for (int i=0; i<N; i++)  {
        myMatrix A_i;
        float u_i = M2d.at(i).at(0);
        float v_i = M2d.at(i).at(1);
        vector<float> M_w = M3d2.at(i);
        vector<float> evenrow;
        // push back 12 numbers in a line one-by-one 
        for(int j=0; j<3; j++){
            evenrow.push_back(M_w.at(j));
        }
        evenrow.push_back(1);
        for(int j=0; j<4; j++){
            evenrow.push_back(0);
        }
        for(int j=0; j<3; j++){
            evenrow.push_back(-M_w.at(j)*u_i);
        }
        evenrow.push_back(-u_i);
        A_i.push_back(evenrow);
    
        vector<float> oddrow;
        for(int k=0; k<4; k++){
            oddrow.push_back(0);
        }
      
        for(int k=0; k<3; k++){
            oddrow.push_back(M_w.at(k));
        }
        oddrow.push_back(1);
        for(int k=0; k<3; k++){
            oddrow.push_back(-M_w.at(k)*v_i);
        }
        oddrow.push_back(-v_i);
        A_i.push_back(oddrow);
        Mat Ai = matFromMatrix(A_i, 0, 2, 12, 0);
        epsilon.push_back(static_cast<float>(norm((Ai * v), NORM_L2)/lamda));
    }
    return epsilon;
    
}

vector<float> inlierPoint(myMatrix M_in, int idx){
    return M_in.at(idx);
}

myMatrix mergeMat(myMatrix m1, myMatrix m2){
    myMatrix m;
    // m1 and m2 must have same columns
    int n = int(m1.size()+m2.size());
    for(int i=0; i<n; i++){
        if(i<int(m1.size())){
            m.push_back(m1.at(i));
        }else{
            m.push_back(m2.at(i-m1.size()));
        }
    }
    return m;
}

void ransac(myMatrix M2d, myMatrix M3d2, int K, int Num){
    float Threshold = 30;
    int max_inliers = 0;
    myMatrix max_tr; // training set with_highest_inliers
    myMatrix m2d_max_tr; // matrix corresponds to the training set
                         // resulting highest inliers
    myMatrix inlierMat;
    myMatrix m2d_inlierMat; // matrix corresponds to the testing set
                            // with all inlier points
    for(int j=0 ; j<200; j++){
        myMatrix resetInlierMat;
        myMatrix reset_m2d_inlierMat;
        int inliers = 0;
        set<int> tr_idx = train_idx(K, Num-1);
        myMatrix train = randMatFromMatrix(M3d2, tr_idx);
        myMatrix M2d_tr = randMatFromMatrix(M2d, tr_idx);

        // Step 2: compute projection matrix P_tr using training points
        Mat P_tr = findProjectionMat(M2d_tr, train, K);
        // displayMat(P_tr);
        set<int> te_idx = test_idx(tr_idx, Num);
        myMatrix test = randMatFromMatrix(M3d2, te_idx);
        myMatrix M2d_te = randMatFromMatrix(M2d, te_idx);
        // // Step 3: for each point in test set, compute projection error
        vector<float> eps = pointProjectionError(M2d_te, test, P_tr);
        for(int i=0; i<int(eps.size()); i++){
            //            cout<<" "<<eps.at(i);
            if(eps.at(i) < Threshold ){
                inliers++;
                resetInlierMat.push_back(inlierPoint(test, i));
                reset_m2d_inlierMat.push_back(inlierPoint(M2d_te, i));
            }

        }

        if(inliers > max_inliers){
            max_inliers = inliers;
            max_tr = train;
            m2d_max_tr = M2d_tr;
            inlierMat = resetInlierMat;
            m2d_inlierMat = reset_m2d_inlierMat;
        }
    }
    int N_calib = max_inliers + int(max_tr.size());
    myMatrix new_m2d = mergeMat(m2d_max_tr, m2d_inlierMat);
    myMatrix new_m3d = mergeMat(max_tr, inlierMat);
    Mat new_P = findProjectionMat(new_m2d, new_m3d, N_calib);
    findParameters(new_P);
}
