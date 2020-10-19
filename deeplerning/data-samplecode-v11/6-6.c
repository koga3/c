#include <stdio.h>

union kanji_byte {
  short i;
  char c[2];
};

int main() {
  union kanji_byte n, m;
  
  printf("漢字一文字を入力してください。");
  scanf("%d", &n.i);

  m.c[0] = n.c[1];
  m.c[1] = n.c[0];
}