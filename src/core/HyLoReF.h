#include <vector>
#include <unordered_map>

using namespace std;

/* Perform the core approach of HyLoReF */
void HyLoReF(double *removedData, double *testData, double *weightData, double *weightData2, int numUser, int numService, int dim, double gamma,
          double lmda, double eta, double eta2, double theta, int maxIter, double alphaInit, double decayRate, int decaySteps, double *Udata,
          double *Sdata, double *lossData, double *errData, bool debugMode);

unordered_map<int, vector<int>> findNeighbors(double **weightMatrix, int numUser);

/* Compute the loss value of HyLoReF */
double loss(double **U, double **S, double **removedMatrix, double **weightMatrix, double **weightMatrix2,
            unordered_map<int, vector<int>> &userNeighbors, unordered_map<int, vector<int>> &serviceNeighbors, double gamma,
            double lmda, double eta, double eta2, double theta, int numUser, int numService, int dim);

/* Compute the gradients of the loss function */
void gradLoss(double **U, double **S, double **removedMatrix, double **weightMatrix, double **weightMatrix2,
              unordered_map<int, vector<int>> &userNeighbors, unordered_map<int, vector<int>> &serviceNeighbors, double **gradU,
              double **gradS, double gamma, double lmda, double eta, double eta2, double theta, int numUser, int numService, int dim);

/* Perform line search to find the best learning rate */
double linesearch(double **U, double **S, double **removedMatrix, double **weightMatrix,
                  unordered_map<int, vector<int>> &userNeighbors,
                  double lastLossValue, double **gradU, double **gradS, double alphaInit, double gamma,
                  double lmda, double eta, double eta2, int numUser, int numService, int dim);

/* Compute predMatrix */
void U_dot_S(double **removedMatrix, double **U, double **S, int numUser,
             int numService, int dim, double **predMatrix);

void errMetric(double **testMatrix, double **U, double **S, int numUser,
               int numService, int dim, double **errResult, int iter);

/* Transform a vector into a matrix */
double **vector2Matrix(double *vector, int row, int col);

/* Compute the dot product of two vectors */
double dotProduct(double *vec1, double *vec2, int len);

/* Allocate memory for a 2D array */
double **createMatrix(int row, int col);

/* Free memory for a 2D array */
void delete2DMatrix(double **ptr);




