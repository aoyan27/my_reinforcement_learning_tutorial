#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define statenum 6
#define goal 5
#define defmu 2.0
#define defsigma 2.0
#define defval 0.0

int actnum=4;
double min=0, max=4;
double Alpha=0.5, Gamma=0.9, r, r1=100, r2=0;
double oldact;
int i,j,k;

/*状態価値、平均、標準偏差のテーブル(状態の数だけ確保されている) */
double val[6]={0,0,0,0,0,0}, mu[6]={2,2,2,2,2,2}, sigma[6]={2,2,2,2,2,2};

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


/* メイン関数*/
int main(void) {

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
