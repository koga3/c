#include <time.h>
#include <math.h>
#include "nn.h"

void print(int m , int n, const float *x)//行数、列数、表示する配列
{
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      printf("  %.5f", x[i + (n * j)]);
    }
    printf("\n");
  }
}

double Uniform( void ){
	return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

float rand_normal( int n ){
	float z=sqrt( -2.0*log(Uniform()) ) * sin( 2.0*3.1415926*Uniform() );
	return 0 + sqrt(sqrt(2.0/784.0)) * z;
 }

void rand_normal_init(int n, float *o)//要素数、配列
{
  for (int i = 0; i < n; i++)
  {
    o[i] = rand_normal(n);
  }
}


int main()
{
  

  // これ以降，
  // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
  // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
  // を使用することができる．

  // ラベルとひらがなの対応は hiragana-labels.txt を見ること

  float A[100];
  rand_normal_init(100, A);
  print(100, 1, A);

  return 0;
}
