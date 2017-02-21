#include <math.h>
#include "mex.h"

const double eps = 0.0000001;

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

mxArray *process(const mxArray *mx_img) 
{	  
  const int iw = 40;
  const int ih = 40;
  const int isz = iw*ih;
  const int pw = 4;
  const int ph = 4;
  
  int feat_dim[3] = {8, 8, 24}; 
  mxArray *mx_feat = mxCreateNumericArray(3, feat_dim, mxSINGLE_CLASS, mxREAL);  
  float *feat = (float*)mxGetPr(mx_feat);  
  float *img = (float*)mxGetPr(mx_img);
  float step = 1.0f / 16.0;
  
  for (int x=pw; x<iw-pw; ++x) {
    for (int y=ph; y<ih-ph; ++y) {
      int idx = y + x*ih;
      float r_val = *(img + idx);
      float g_val = *(img + idx + isz);
      float b_val = *(img + idx + 2*isz);
      
      int i = (y-ph) / ph;
      int j = (x-pw) / pw;
      int k_r = (int)(floor(r_val / 32.0));
      int k_g = (int)(floor(g_val / 32.0)) + 8;
      int k_b = (int)(floor(b_val / 32.0)) + 16;
      
      *(feat + k_r*64 + j*8 + i) += step;
      *(feat + k_g*64 + j*8 + i) += step;
      *(feat + k_b*64 + j*8 + i) += step;      
    }
  }
  
	return mx_feat;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {  
  plhs[0] = process(prhs[0]); 
}
