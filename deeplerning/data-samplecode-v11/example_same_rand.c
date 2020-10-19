#include <time.h>
#include "nn.h"

//配列の行列表示
/*void print(int m , int n, const float *x)//行数、列数、表示する配列
{
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      printf("  %.4f", x[i + (n * j)]);
    }
    printf("\n");
  }
}*/

//FC層
//行列とベクトルの積
void mul(int m, int n, const float *x, const float *A, float *o)//行数、列数、ベクトル、行列、積の値
{
  for (int j = 0; j < m; j++)
  {
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
      sum += (float)(A[(j * n) + i] * x[i]);
    }
    o[j] = sum;
  }
}
//配列（行列、ベクトル）の和
void add(int m, const float *x, float *o)//配列の要素数、足すやつ、足されるやつ
{
  for (int i = 0; i < m; i++)
  {
    o[i] += x[i];
  }
}
//ｆｃの計算
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)//Aの行数、列数、入力、A、ｂ、出力
{
  mul(m, n, x, A, y);
  add(m, b, y);
}
//FC層

//ReLU層
void relu(int n, const float *x, float *y)//配列の要素数、入力、出力
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
//ReLU層

//ソフトマックス層
//配列の最大値を出力
float array_max(int n, const float *x)//配列の要素数、入力
{
  float range = 0;
  for (int i = 0; i < 10; i++)
  {
    if (range < x[i])
    {
      range = x[i];
    }
  }
  return range;//配列の最大値
}
//ソフトマックスの計算
void softmax(int n, const float *x, float *y)//配列の要素数、入力、出力
{
  float xmax = array_max(n, x);
  float sumexp = 0;
  for (int i = 0; i < n; i++)
  {
    sumexp += expl(x[i] - xmax);
  }

  for (int i = 0; i < n; i++)
    y[i] = expl(x[i] - xmax) / sumexp;
}
//ソフトマックス層

//入力それたパラメータによる推論（6層）
int inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x)//fc層のパラメータ、入力
{
  float fc_x[784] = {};
  for (int i = 0; i < 784; i++)
    fc_x[i] = x[i];
  float y1[50], y2[100], y3[10];
  fc(50, 784, fc_x, A1, b1, y1);
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
  return ans;//推論結果
}


//誤差逆伝搬
//softmax層
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)//softmax層の出力の要素数、softmax層の出力、正解、勾配
{
  float vc_t[10] = {};
  vc_t[t] = 1.0;
  for (int i = 0; i < n; i++)
    dEdx[i] = y[i] - vc_t[i];
}
//relu層
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx)//要素数、relu層への入力、上流側からの勾配、下流側への勾配
{
  for (int i = 0; i < n; i++)
  {
    if (x[i] > 0)
      dEdx[i] = dEdy[i];
    else
      dEdx[i] = 0;
  }
}

void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A, float *dEdA, float *dEdb, float *dEdx)//Aの行数、列数、fc層への入力、上流側からの勾配、A、Aの勾配、ｂの勾配、下流側への勾配
{
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
      dEdA[i + (j * n)] = (float)(dEdy[j] * x[i]);
    dEdb[j] = dEdy[j];
  }

  for (int k = 0; k < n; k++)
  {
    float sum = 0;
    for (int j = 0; j < m; j++)
      sum += (float)(A[k + (j * n)] * dEdy[j]);
    dEdx[k] = sum;
  }
}

//誤差逆伝搬
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t, float *y, float *dEdA1, float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3)//各種パラメータ、NNへの入力、正解、出力、パラメータの勾配
{
  //必要な入力を記憶しながら推論
  float relu1_x[50], fc2_x[50], relu2_x[100], fc3_x[100];//勾配の計算で必要な入力
  //入力の保存
  float fc1_x[784];
  for (int i = 0; i < 784; i++)
    fc1_x[i] = x[i];
  //推論
  fc(50, 784, fc1_x, A1, b1, relu1_x);
  relu(50, relu1_x, fc2_x);
  fc(100, 50, fc2_x, A2, b2, relu2_x);
  relu(100, relu2_x, fc3_x);
  fc(10, 100, fc3_x, A3, b3, y);
  softmax(10, y, y);

  //勾配を計算
  float dEdx_fc3[10];//fc3への勾配
  softmaxwithloss_bwd(10, y, t, dEdx_fc3);

  float dEdx_relu2[100];//reru2
  fc_bwd(10, 100, fc3_x, dEdx_fc3, A3, dEdA3, dEdb3, dEdx_relu2);

  float dEdx_fc2[100];//fc2
  relu_bwd(100, relu2_x, dEdx_relu2, dEdx_fc2);

  float dEdx_relu1[50];//relu1
  fc_bwd(100, 50, fc2_x, dEdx_fc2, A2, dEdA2, dEdb2, dEdx_relu1);

  float dEdx_fc1[50];//fc1
  relu_bwd(50, relu1_x, dEdx_relu1, dEdx_fc1);

  float dEdx[784];//勾配
  fc_bwd(50, 784, fc1_x, dEdx_fc1, A1, dEdA1, dEdb1, dEdx);
}

//配列のランダムシャフル
void swap_i(int *pa, int *pb)//入力二つの値を交換する
{
  int temp = *pa;
  *pa = *pb;
  *pb = temp;
}
void shuffle(int n, int *x)//配列の要素数、入出力
{
  for (int i = 0; i < n; i++)
  {
    int j = rand() % n;
    swap_i(&x[i], &x[j]);
  }
}

//損失関数
float cross_entropy_error(const float *y, int t)//NNの出力、正解
{
  return -logf(y[t] + (float)1e-7);
}

//補助関数
//配列（行列、ベクトル）のスカラー積
void scale(int n, float x, float *o)//配列の要素数、スカラー、配列
{
  for (int i = 0; i < n; i++)
    o[i] *= x;
}
//配列の初期化
void init(int n, float x, float *o)//要素数、初期化する値、配列
{
  for (int i = 0; i < n; i++)
    o[i] = x;
}
//配列の[-1:1]でのランダムな初期化
void rand_init(int n, float *o)//要素数、配列
{
  for (int i = 0; i < n; i++)
  {
    float y = (float)(rand() * 2.0 / RAND_MAX);
    o[i] = y - 1.0;
  }
}

//パラメータの保存
void save(const char *filename, int m, int n, const float *A, const float *b)//保存先、Aの行数、列数、A、ｂ
{
  FILE *fp;
  fp = fopen(filename, "wb+");
  fwrite(A, sizeof(float), m * n, fp);
  fwrite(b, sizeof(float), m, fp);
  fclose(fp);
}

//main関数
int main()
{
  srand(time(NULL));
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;
  float *test_x = NULL;
  unsigned char *test_y = NULL;
  int test_count = -1;
  int width = -1;
  int height = -1;
  load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);


  //定数の定義
  const static int epoc_count = 10;//エポック数
  const static int minipatch_size = 100;//ミニパッチサイズ
  const static float lerate = 0.1;//学習率

  //パラメータのメモリ確保
  float *A1 = malloc(sizeof(float) * 784 * 50), *A2 = malloc(sizeof(float) * 50 * 100), *A3 = malloc(sizeof(float) * 100 * 10);
  float *b1 = malloc(sizeof(float) * 50), *b2 = malloc(sizeof(float) * 100), *b3 = malloc(sizeof(float) * 10);

  //パラメータの初期化
  rand_init(784 * 50, A1);
  rand_init(50, b1);
  rand_init(50 * 100, A2);
  rand_init(100, b2);
  rand_init(100 * 10, A3);
  rand_init(10, b3);

  //配列indexの初期化
  int *index = malloc(sizeof(int) * train_count);
  for (int i = 0; i < train_count; i++)
    index[i] = i;

  //エポック
  int k = 0;
  float correst_rate_test, correst_rate_train;
  do  
  {
    //indexのシャッフル
    shuffle(train_count, index);

    //ミニパッチ学習
    for (int i = 0; i < (train_count / minipatch_size); i++)
    {
      //平均勾配の初期化
      float avg_dEdA1[784 * 50] = {}, avg_dEdA2[50 * 100] = {}, avg_dEdA3[100 * 10] = {};
      float avg_dEdb1[50] = {}, avg_dEdb2[100] = {}, avg_dEdb3[10] = {};
      float *y = malloc(sizeof(float) * 10);

      //n個indexから取り出す
      int *index_n = malloc(sizeof(int) * minipatch_size);
      for (int j = 0; j < minipatch_size; j++)
        index_n[j] = index[j + (i * minipatch_size)];
      //平均勾配の計算
      for (int j = 0; j < minipatch_size; j++)
      {
        //勾配の定義・初期化
        float dEdA1[784 * 50] = {}, dEdb1[50] = {}, dEdA2[50 * 100] = {}, dEdb2[100] = {}, dEdA3[100 * 10] = {}, dEdb3[10] = {};
        //誤差逆伝搬
        backward6(A1, b1, A2, b2, A3, b3, train_x + width * height * index_n[j], train_y[index_n[j]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);
        //勾配を足す
        add(784 * 50, dEdA1, avg_dEdA1);
        add(50 * 100, dEdA2, avg_dEdA2);
        add(100 * 10, dEdA3, avg_dEdA3);
        add(50, dEdb1, avg_dEdb1);
        add(100, dEdb2, avg_dEdb2);
        add(10, dEdb3, avg_dEdb3);
      }
      //ミニパッチサイズで割って、平均を求める
      scale(784 * 50, (float)(1.0 / minipatch_size), avg_dEdA1);
      scale(50 * 100, (float)(1.0 / minipatch_size), avg_dEdA2);
      scale(100 * 10, (float)(1.0 / minipatch_size), avg_dEdA3);
      scale(50, (float)(1.0 / minipatch_size), avg_dEdb1);
      scale(100, (float)(1.0 / minipatch_size), avg_dEdb2);
      scale(10, (float)(1.0 / minipatch_size), avg_dEdb3);

      //A,bの更新
      scale(784 * 50, -lerate, avg_dEdA1);
      scale(50 * 100, -lerate, avg_dEdA2);
      scale(100 * 10, -lerate, avg_dEdA3);
      scale(50, -lerate, avg_dEdb1);
      scale(100, -lerate, avg_dEdb2);
      scale(10, -lerate, avg_dEdb3);
      add(784 * 50, avg_dEdA1, A1);
      add(50 * 100, avg_dEdA2, A2);
      add(100 * 10, avg_dEdA3, A3);
      add(50, avg_dEdb1, b1);
      add(100, avg_dEdb2, b2);
      add(10, avg_dEdb3, b3);
    }

    //損失関数の計算
    float E = 0;//損失関数
    for (int i = 0; i < test_count; i++)
    {
      //テストデータの推論
      float y1[50], y2[100], y3[10];
      fc(50, 784, test_x + i * 784, A1, b1, y1);
      relu(50, y1, y1);
      fc(100, 50, y1, A2, b2, y2);
      relu(100, y2, y2);
      fc(10, 100, y2, A3, b3, y3);
      softmax(10, y3, y3);
      //損失関数の計算
      E += cross_entropy_error(y3, test_y[i]);
    }
    E /= test_count;
    printf("epoc.%d  E=%f", k + 1, E);

    //正解率の計算
    //test
    int sum = 0;//正解数
    for (int i = 0; i < test_count; i++)
    {
      if (inference6(A1, b1, A2, b2, A3, b3, test_x + i * width * height) == test_y[i])
        sum++;
    }
    correst_rate_test = (float)(sum * 100.0 / test_count);
    printf("  %f%%", correst_rate_test);
    //taining
    int sum1 = 0;//正解数
    for (int i = 0; i < train_count; i++)
    {
      if (inference6(A1, b1, A2, b2, A3, b3, train_x + i * width * height) == train_y[i])
        sum1++;
    }
    correst_rate_train = (float)(sum1 * 100.0 / train_count);
    printf("  %f%%\n\n", correst_rate_train);
    k++;
  } while ((k < epoc_count)&&(correst_rate_train - correst_rate_train<(float)10.0));

  //パラメータの保存
  save("fc1.dat", 50, 784, A1, b1);
  save("fc2.dat", 100, 50, A2, b2);
  save("fc3.dat", 10, 100, A3, b3);

  return 0;
}
