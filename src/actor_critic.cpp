#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define statenum 6
#define goal 5
#define defmu 2.0
#define defsigma 2.0
#define defval 0.0

double val[6]={0,0,0,0,0,0}, mu[6]={2,2,2,2,2,2}, sigma[6]={2,2,2,2,2,2};
int actnum=4;
double min=0, max=4;
double Alpha=0.5, Gamma=0.9, r, r1=100, r2=0;
double oldact;
int i,j,k;

/*環境 (数は部屋番号)*/
int status[6][4] ={{0,1,3,0},{1,2,4,0},{2,2,5,1},{0,4,3,3},{1,5,4,3},{0,0,0,0}};


/*乱数発生*/
double NumR() {/* 0~1一様乱数 */
	return (double)rand()/((double)RAND_MAX+1);
}


double BoxMuller(double mean, double var) {/*正規 mean:平均,var:分散 Box-Muller法*/
	double r1, r2, z1, z2;
	r1 = rand()/(double)RAND_MAX;
	// printf("r1 = %f\n", r1);
	r2 = rand()/(double)RAND_MAX;
	// printf("r2 = %f\n", r2);
	z1 = sqrt(-2.0*log(r1));
	z2 = sin(6.283185307179586*r2);
	return var*z1*z2 + mean;
}

/* アクター */
int Aaction(int state){
	double act;
	act = BoxMuller(mu[state],sigma[state]);
	oldact = act;
	if(act<min){
		do{
			// printf("act_min(before) = %lf\n", act);
			act = act + actnum;
			// printf("act_min(after) = %lf\n", act);
		}while(act<min);
	}
	if(act>=max){
		do{
			// printf("act_max(before) = %lf\n", act);
			act = act-actnum;
			// printf("act_max(after) = %lf\n", act);
		}while(act>=max);
	}
	return (int)act;
}


//次の状態を現在の状態とアクションから導く
int nextStatus(int now,int action){
	int next;
	if(now == goal){ 
		next=0;
	}
	else{
		next = status[now][action];
	}
	return(next);
}


//標準偏差と平均の更新（正規分布の更新）
void Arenew(int before){
	printf("sigma[%d](before) = %f\n", before, sigma[before]);
	sigma[before] = (sigma[before]+fabs(oldact-mu[before]))/2;
	printf("sigma[%d](after) = %f\n", before, sigma[before]);
	printf("mean[%d](before) = %f\n", before, mu[before]);
	mu[before] = (mu[before]+oldact)/2;
	printf("mean[%d](after) = %f\n", before, mu[before]);
	// if(fabs(oldact-mu[before])<sigma[before]){
	// }
	if(oldact<mu[before]){
		do{
			mu[before] = mu[before] + actnum;
		}while(mu[before]<min);
	}
	if(mu[before]>=max){
		do{
			mu[before] = mu[before]-actnum;
		}while(mu[before]>=max);
	}
}


/* クリティック*/
void Grenew(int now,int before){
	double tderror;
	if(now == goal) r = 100.0;
	else r = 0.0;
	//TD-error を求める
	tderror = r + Gamma*val[now]-val[before];
	printf("TDerror = %f\n", tderror);
	//状態評価値の更新
	val[before] = val[before]+Alpha*tderror;
	printf("Val[%d] = %f\n", before, val[before]);
	//TD 誤差を返す
	if(tderror>0){
		Arenew(before);
		// printf("TDerror > 0\n");
	}
}

/*
void report() {
	int i;
	for(i=0; i<statenum; i++) {
		printf("%d:%f==%f_%f¥n",i,val[i],mu[i],sigma[i]);
	}
}*/
/*
void print_histogram(FILE **fp, double list[100000]){
	int count = 0;
	for(int j=0;j<2000;j++){
		for(int i=0;i<100000;i++){
			// printf("list[%d] = %f\n", i,list[i]);
			if(list[i]>(-1.0+0.001*j) && list[i]<(-0.9+0.001*j)){
				count += 1;
			}
		}
		fprintf(*fp,"%lf, %d\n",(-0.9+0.001*j), count);
		count = 0;
	}	
}*/


/* メイン関数*/
int main(void) {
	// FILE *fp;
	// const char *fname = "box_muller.csv";
	// fp = fopen(fname, "w");
	// if(fp == NULL){
		// printf("Can not open %s!!", fname);
		// return -1;
	// }
	
	// double *list = (double *)malloc(sizeof(double)*100000);

	// printf("rand() = %f\n", NumR());
	// printf("rand_max = %d\n", RAND_MAX);
	// for(int i=0;i<100000;i++){
		// printf("box_muller = %f\n", BoxMuller(0.0,1.0));
		// list[i] = BoxMuller(0.0, 1.0);
		// printf("list[%d] = %f\n", i, list[i]);
		// fprintf(fp, "%f,\n", BoxMuller(0.0,1.0));
	// }
	
	// print_histogram(&fp, list);	
	// free(list);


	// fclose(fp);

	int before=0, now=0;
	int count=0, nowt;
	// report();
	while(count<3000){
		while (now != goal){
			printf("count = %d\n", count);
			before=now;
			printf("before = %d\n", before);
			nowt= Aaction(before);
			printf("nowt = %d\n", nowt);
			now=nextStatus(now, nowt);
			printf("now = %d\n", now);
			Grenew(now,before);
			printf("\n");
		}

		for(int i=0;i<6;i++){
			printf("mu[%d] = %f\n", i, mu[i]);
		}
		for(int i=0;i<6;i++){
			printf("sigma[%d] = %f\n", i, sigma[i]);
		}
		for(int i=0;i<6;i++){
			printf("val[%d] = %f\n", i, val[i]);
		}
		count++;
		now=0;
	}
	// report();
	return 0;
}
