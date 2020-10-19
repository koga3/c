#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

int main()
{
  int n, x, fin = 0, omikuji = 0;
  srand(time(NULL));

  printf("You can do Janken in this application. And this also has Omikji.\n You can chose 0:Block, 2:Scissor or 5:Paper.\n\n");
  while (fin != 1)
  {
    omikuji = 0;
    /* コンピュータの選択 */
    switch (rand() % 3)
    {
    case 0:
      x = 0;
      break;
    case 1:
      x = 2;
      break;
    case 2:
      x = 5;
      break;
    }
    /* コンピュータの選択 */

    /* ユーザの選択 */
    printf("Your input (0,2,5):");
    scanf("%d", &n);
    /* ユーザの選択 */

    switch (n)
    {
    /* ユーザが0 */
    case 0:
      switch (x)
      {
      case 0:
        printf("Comp:0 vs You:0 => Try again.\n");
        break;
      case 2:
        printf("Comp:2 vs You:0 => You win.\n");
        fin = 1;
        omikuji = (rand() % 100) + 1;
        break;
      case 5:
        printf("Comp:5 vs You:0 => You lose.\n");
        fin = 1;
        omikuji = -1;
        break;
      }
      break;
    /* ユーザが0 */
    /* ユーザが2 */
    case 2:
      switch (x)
      {
      case 0:
        printf("Comp:0 vs You:2 => You lose.\n");
        fin = 1;
        omikuji = -1;
        break;
      case 2:
        printf("Comp:2 vs You:2 => Try again.\n");
        break;
      case 5:
        printf("Comp:5 vs You:2 => You win.\n");
        fin = 1;
        omikuji = (rand() % 100) + 1;
        break;
      }
      break;
    /* ユーザが2 */
    /* ユーザが5 */
    case 5:
      switch (x)
      {
      case 0:
        printf("Comp:0 vs You:5 => You win.\n");
        fin = 1;
        omikuji = (rand() % 100) + 1;
        break;
      case 2:
        printf("Comp:2 vs You:5 => You lose.\n");
        fin = 1;
        omikuji = -1;
        break;
      case 5:
        printf("Comp:5 vs You:5 => Try again.\n");
        break;
      }
      break;
    /* ユーザが5 */
    default:
      printf("Invalid input => Input again.\n");
      break;
    }

    if (omikuji == 0)
    {
      printf("\n");
    }
    else if (omikuji == -1)
    {
      printf("\n\n\nKyo\n\n\n\n");
    }
    else if (omikuji == 100)
    {
      printf("\n\n\nDaikichi!!\n\n\n\n");
    }
    else if ((omikuji % 10) < 6)
    {
      printf("\n\n\nKichi\n\n\n\n");
    }
    else
    {
      printf("\n\n\nChukichi\n\n\n\n");
    }

    int yn;

    while (fin == 1)
    {
      printf("Try again? 1:Yes 0:No\n");
      scanf("%d", &yn);

      if (yn == 1)
      {
        fin = 0;
        break;
      }
      else if (yn == 0)
      {
        break;
      }
    }
  }

  return 0;
}