#include "pwm.h"
 
#define PWM0_CFG (__METAL_ACCESS_ONCE((__metal_io_u32 *)(METAL_SIFIVE_PWM0_0_BASE_ADDRESS + METAL_SIFIVE_PWM0_PWMCFG)))
#define PWM0_CMP0 (__METAL_ACCESS_ONCE((__metal_io_u32 *)(METAL_SIFIVE_PWM0_0_BASE_ADDRESS + METAL_SIFIVE_PWM0_PWMCMP0)))
#define PWM0_CMP1 (__METAL_ACCESS_ONCE((__metal_io_u32 *)(METAL_SIFIVE_PWM0_0_BASE_ADDRESS + METAL_SIFIVE_PWM0_PWMCMP1)))
#define PWM0_CMP2 (__METAL_ACCESS_ONCE((__metal_io_u32 *)(METAL_SIFIVE_PWM0_0_BASE_ADDRESS + METAL_SIFIVE_PWM0_PWMCMP2)))
#define PWM0_CMP3 (__METAL_ACCESS_ONCE((__metal_io_u32 *)(METAL_SIFIVE_PWM0_0_BASE_ADDRESS + METAL_SIFIVE_PWM0_PWMCMP3)))

int pwm0_setcmp(int cmp_num,char cmp){
	if(cmp_num==0){
		PWM0_CMP0 &=0;
		PWM0_CMP0|=cmp;
	}
	else if(cmp_num==1){
		PWM0_CMP1 &=0;
		PWM0_CMP1|=cmp;
	}
	else if(cmp_num==2){
		PWM0_CMP2 &=0;
		PWM0_CMP2|=cmp;
	}
	else if(cmp_num==3){
		PWM0_CMP3 &=0;
		PWM0_CMP3|=cmp;
	}
}

void pwm_init(int scale){

	//GPIO2->PWM0CMP2
	GPIO0_IOF_EN |=(1<<2);//enable gpio2 iof
	GPIO0_IOF_SEL |=(1<<2); //select iof 1

	PWM0_CFG &=0;//clear cfg
	PWM0_CFG |=scale;//set scale=0
	PWM0_CFG |=(1<<12);//set pwmenalways 1 ;note:this must be done before pwmzeorcmp
	PWM0_CFG |=(1<<9);//set pwmzero 1

	//init the cmpx value
	pwm0_setcmp(0,250); 
	pwm0_setcmp(1,250); 
	pwm0_setcmp(2,250); 
	pwm0_setcmp(3,250); 
}
