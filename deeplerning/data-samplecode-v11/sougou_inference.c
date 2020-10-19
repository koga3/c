#include "nn.h"


/*ここからlerningファイルと同じ*/
void mul(int m, int n, const float *x, const float *A, float *o)
{
  for (int j = 0; j < m; j++)
  {
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
      sum += A[(j * n) + i] * x[i];
    }
    o[j] = sum;
  }
}

void add(int m, const float *x, float *o)
{
  for (int i = 0; i < m; i++)
  {
    o[i] += x[i];
  }
}


void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
  mul(m, n, x, A, y);
  add(m, b, y);
}

void relu(int n, const float *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    if (x[i] <= 0)
    {
      y[i] = 0;
    }
    else
    {
      y[i] = x[i];
    }
  }
}

float array_max(int n, const float *x)
{
  float range = 0;
  for (int i = 0; i < 10; i++)
  {
    if (range < x[i])
    {
      range = x[i];
    }
  }
  return range;
}

void softmax(int n, const float *x, float *y)
{
  float xmax = array_max(n, x);
  float sumexp = 0;
  for (int i = 0; i < n; i++)
  {
    sumexp += expl(x[i] - xmax);
  }

  for (int i = 0; i < n; i++)
  {
    y[i] = expl(x[i] - xmax) / sumexp;
  }
}

int inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x)
{
  float y1[50], y2[100], y3[10];
  fc(50, 784, x, A1, b1, y1);
  relu(50, y1, y1);
  fc(100, 50, y1, A2, b2, y2);
  relu(100, y2, y2);
  fc(10, 100, y2, A3, b3, y3);
  softmax(10, y3, y3);
  //Output
  float range = 0;
  int ans = 0;
  for (int i = 0; i < 10; i++)
  {
    if (range < y3[i])
    {
      range = y3[i];
      ans = i;
    }
  }
  return ans;
}
/*ここまでlerningファイルと同じ*/

//指定したファイルから一層分のパラメータをロード
void load(const char *filename, int m, int n, float *A, float *b)//ロードするファイル名、Aの行数、列数、A、ｂ
{
  FILE *fp;
  fp = fopen(filename, "rb+");
  if (!fp)
    printf("File cannot open");
  fread(A, sizeof(float), m * n, fp);
  fread(b, sizeof(float), m, fp);
  fclose(fp);
}

//パラメータを用いて、画像ファイルを推論
int main(int argc, char *argv[])//画像ファイルとパラメータのファイル三種を入力
{
  //メモリを確保
  float *A1 = malloc(sizeof(float) * 784 * 50);
  float *b1 = malloc(sizeof(float) * 50);
  float *A2 = malloc(sizeof(float) * 50 * 100);
  float *b2 = malloc(sizeof(float) * 100);
  float *A3 = malloc(sizeof(float) * 100 * 10);
  float *b3 = malloc(sizeof(float) * 10);
  //画像データをロード
  float *x = load_mnist_bmp(argv[4]);
  //パラメータをロード
  load(argv[1], 50, 784, A1, b1);
  load(argv[2], 100, 50, A2, b2);
  load(argv[3], 10, 100, A3, b3);

  //推論結果を表示
  printf("Answer: %d", inference6(A1, b1, A2, b2, A3, b3, x));
  return 0;
}