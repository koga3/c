#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

int main()
{
  srand(time(NULL));
  int hl;
  int wl;
  int money;
  int bet;

  /*持ち金の設定*/
  printf("How much money do you use?\n");
  scanf("%d", &money);
  /*持ち金の設定*/

  while (hl != 2)
  {
    /*かけ金の設定*/
    printf("How much money do you bet?\n");
    scanf("%d", &bet);
    money -= bet;
    /*かけ金の設定*/

    /*コンピュータのトランプ*/
    int x = (rand() % 13) + 1;
    printf("PC's card is [%d]\n", x);
    /*コンピュータのトランプ*/

    /*ユーザの入力*/
    printf("high : 1     low : 0\n");
    scanf("%d", &hl);
    /*ユーザの入力*/

    /*勝敗の決定*/
    int n = (rand() % 12) + 1;
    switch (hl)
    {
    case 1:
      if (n > x)
      {
        printf("You win!!\n\n\n");
        wl = 1;
      }
      else if (n < x)
      {
        printf("You lose...\n\n\n");
        wl = 0;
      }
      else
      {
        printf("Draw.\n\n\n");
        wl = 2;
      }
      printf("Your card is [%d]\n", n);
      /*勝敗の決定*/
      break;
          case 0:
      if (n < x)
      {
        printf("You win!!\n\n\n");
        wl = 1;
      }
      else if (n > x)
      {
        printf("You lose...\n\n\n");
        wl = 0;
      }
      else
      {
        printf("Draw.\n\n\n");
        wl = 2;
      }
      printf("Your card is [%d]\n", n);
      /*勝敗の決定*/
      break;
      default:
        printf("Error");
        break;
    }

    switch (wl)
    {
    case 1:
      bet *= 2;
      break;
    case 0:
      bet *= 0;
      break;
    case 2:
      break;
    }
    money += bet;
    printf("Your money is %d now.\n\n\n", money);
  }

  return 0;
}