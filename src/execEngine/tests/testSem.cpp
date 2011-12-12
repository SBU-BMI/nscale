#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

int main(){
	sem_t* sem_name;

	sem_name = sem_open("bla.txt", (O_CREAT|O_EXCL), 0644,1);
 
	if( sem_name== SEM_FAILED ){
		printf("I'm sorry bro!\n");
		exit(0);
	}

//	sem_init(sem_name, 0, 10);

	int value; 
	sem_getvalue(sem_name, &value); 
	printf("The value of the semaphors is %d\n", value);


	return 0;
}

