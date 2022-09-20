#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include "BIT_OPS.h"
#include "RTC.h"
#include <math.h>


/*DATA STRUCT*/
struct Measurement{
    int distance;
    int angle;
};

/*PIN DEFINES*/
#define TRIG_PIN 1		// Trigger pin at GPIO 1 (pin 9) (OUTPUT)
#define ECHO_PIN 9		// Echo pin at GPIO 9 (pin 15) (INPUT)
#define SERVO_PIN 11	// Servo pin at GPIO 11 (pin 17) (OUTPUT)

/*FUNCTION DECLARATIONS*/
void setInputPin(uint32_t pinNumber);
void setOutputPin(uint32_t pinNumber);
void setPWMPin(uint32_t pinNumber);
void setPWM1(uint32_t frequency, uint32_t dutyCycle);
void setDutyCycle1(uint32_t dutyCycle);
void resetSingleShot();
void setPWM2(uint32_t frequency, uint32_t dutyCycle);
void setDutyCycle2(uint32_t dutyCycle);
void setSingleShotPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle);
void setContinuousPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle);
void save_measurement(int time, int angle);
void stream_output(int distance, int angle);
void bubble_sort(struct Measurement *array, int array_size);
void delay_us(uint32_t delay_in_us);
uint32_t read_echo();


/*Global VARIABLES*/
#define ARRAY_MAX_SIZE 65
static uint32_t BASE_FREQUENCY = 16000000;
struct Measurement measurements [ARRAY_MAX_SIZE] = {}; // Up to 65 measurements in the array
static uint32_t array_element_index = 0;
uint32_t motor_angle = 50;
static uint32_t speed_of_sound = 340; // cm/s
static uint32_t sensor_timeout = 25000;  // Units ?
static uint32_t loopTime = 60000;


/*REGISTRY DECLARATIONS*/
volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;		// Registry for INPUT values
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;		// Registry for INPUT set pin ENABLE
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;		// Registry for OUTPUT values
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;		// Registry for OUTPUT set pin ENALBE
volatile uint32_t *GPIO_PUE 		= (uint32_t*)0x10012010;		// Registry for INPUT Internal PULL UP ENABLE
volatile uint32_t *IOF_EN 			= (uint32_t*)0x10012038;		// Registry for HW-Driven funtions (IOF)
volatile uint32_t *IOF_SEL 			= (uint32_t*)0x1001203C;		// Registry for HW-Driven funtions (IOF)

volatile uint32_t *PWM0_1_CONFIG	= (uint32_t*)0x10015000;		// Registry for PWM CONFIG TRIGGER PIN
volatile uint32_t *PWM0_1_COUNT		= (uint32_t*)0x10015008;		// Registry for PWM Counter TRIGGER PIN
volatile uint32_t *PWM0_1_CMP0		= (uint32_t*)0x10015020;		// Registry for PWM COMPERATOR TRIGGER PIN
volatile uint32_t *PWM0_1_CMP1		= (uint32_t*)0x10015024;		// Registry for PWM COMPERATOR TRIGGER PIN
volatile uint32_t *PWM0_1_CMP2		= (uint32_t*)0x10015028;		// Registry for PWM COMPERATOR TRIGGER PIN
volatile uint32_t *PWM0_1_CMP3		= (uint32_t*)0x1001502C;		// Registry for PWM COMPERATOR TRIGGER PIN

volatile uint32_t *PWM2_1_CONFIG	= (uint32_t*)0x10035000;		// Registry for PWM CONFIG SERVO PIN
volatile uint32_t *PWM2_1_COUNT		= (uint32_t*)0x10035008;		// Registry for PWM Counter SERVO PIN
volatile uint32_t *PWM2_1_CMP0		= (uint32_t*)0x10035020;		// Registry for PWM COMPERATOR SERVO PIN
volatile uint32_t *PWM2_1_CMP1		= (uint32_t*)0x10035024;		// Registry for PWM COMPERATOR SERVO PIN
volatile uint32_t *PWM2_1_CMP2		= (uint32_t*)0x10035028;		// Registry for PWM COMPERATOR SERVO PIN
volatile uint32_t *PWM2_1_CMP3		= (uint32_t*)0x1003502C;		// Registry for PWM COMPERATOR SERVO PIN


int main() {
	bool reverse = false;
	setInputPin(ECHO_PIN);
	//setOutputPin(TRIG_PIN);	// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)
	setSingleShotPWM(TRIG_PIN,20000,50);
	setContinuousPWM(SERVO_PIN, 700, 100);
	rtc_setup();

	// Initialize servo
	// Move servo to starting position 1 / 65

	while(1) {
		/*
		printf("\r COUNT: %u\n", *PWM2_1_COUNT);
		printf("\r COMP 0: %u\n", *PWM2_1_CMP0);
		printf("\r COMP 1: %u\n", *PWM2_1_CMP1);
		printf("\r CONFIG: %u\n", *PWM2_1_CONFIG);
		*/
		uint32_t waitTime = get_rtc_low_micro() + 100000;
		while(get_rtc_low_micro() <= waitTime);
		if(reverse){
			motor_angle --;
		}
		else if(!reverse){
			motor_angle ++;
		}
		setDutyCycle2(motor_angle);
		if(motor_angle >= 99){
			reverse = true;
		}
		else if(motor_angle <= 40){
			reverse = false;
		}
		// Trigger sensor
		resetSingleShot();
		uint32_t distance = (read_echo() * speed_of_sound)/2000;

		printf("\r Distance in: %u\n", distance);
		printf("\r MotorAngle: %u\n", motor_angle);

		// Measurement should only be saved if value > 0
		// save_measurement(end_time-start_time, motor_angle);

		// Print results in terminal
		// stream_output(int distance, int angle);

		// Move motor to next position here

		// If measurements are over


		/*
		Print all array distances:
		for(int i=0; i<array_element_index; i++){
            	printf("Array distance[%d]: %d\n", i, measurements[i].distance);
        	}
		Sort the array:
		bubble_sort(measurements, array_element_index);
		*/
	}
}

void setInputPin(uint32_t pinNumber){
	*IOF_EN 		&= ~(1<<pinNumber);
	*GPIO_OUTPUT_EN &= ~(1<<pinNumber);
	*GPIO_INPUT_EN 	|= (1<<pinNumber);
	*GPIO_PUE 		|= (1<<pinNumber);
}

void setOutputPin(uint32_t pinNumber){
	*IOF_EN 		&= ~(1<<pinNumber);
	*GPIO_INPUT_EN 	&= ~(1<<pinNumber);
	*GPIO_PUE 		&= ~(1<<pinNumber);
	*GPIO_OUTPUT_EN |= (1<<pinNumber);

}

void setPWMPin(uint32_t pinNumber){
	*GPIO_INPUT_EN 	&= ~(1<<pinNumber);
	*GPIO_PUE 		&= ~(1<<pinNumber);
	*GPIO_OUTPUT_EN &= ~(1<<pinNumber);
	*GPIO_INPUT_EN 	&= ~(1<<pinNumber);
	*IOF_EN 		|= (1<<pinNumber);
	*IOF_SEL		|= (1<<pinNumber);
}

void setDutyCycle1(uint32_t dutyCycle){
	*PWM0_1_CMP1 =((*PWM0_1_CMP0 * (100 - dutyCycle))/100);
}

void setPWM1(uint32_t frequency, uint32_t dutyCycle){
	int maxComparator =  256 - 1; // Maximum 16-bit comparator value
	//int scale = 0;
	uint32_t comparator; //= BASE_FREQUENCY / (frequency * pow(2,scale));

	int scale = -1;

	do
	{
		scale++;
		comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));
		printf("\r PWM1 Comparator: %u\n", comparator);
		printf("\r PWM1 SCALE: %u\n", scale);
	} while (comparator > maxComparator);
	/*
	while(comparator > maxComparator){
		scale += 1;
		comparator = BASE_FREQUENCY / (frequency * pow(2,scale));
		if(comparator <= maxComparator) {
			scale -= 1;
			printf("\r SCALE Break: %u\n", scale);
			break;
		}
		printf("\r SCALE: %u\n", scale);
	}
	*/
	*PWM0_1_CMP0 = 0b0000000000000000;
	*PWM0_1_CMP0 |= comparator;
	*PWM0_1_CONFIG |= scale;
	*PWM0_1_CMP1 = 0b0000000000000000;
	*PWM0_1_CMP1 |= (comparator * ((100 - dutyCycle))/100);
	printf("\r PWM1 Comparator0 Final: %u\n", *PWM0_1_CMP0);
	printf("\r PWM1 Comparator1 Final: %u\n", *PWM0_1_CMP1);
	printf("\r PWM1 SCALE FINAL: %u\n", scale);
	printf("\r PWM1 CONFIG FINAL: %u\n", *PWM0_1_CONFIG);
}

void setDutyCycle2(uint32_t dutyCycle){
	*PWM2_1_CMP1 =((*PWM2_1_CMP0 * (100 - dutyCycle))/100);
}

void setPWM2(uint32_t frequency, uint32_t dutyCycle){
	int maxComparator =  65536 - 1; // Maximum 16-bit comparator value
	//int scale = 0;
	uint32_t comparator; //= BASE_FREQUENCY / (frequency * pow(2,scale));

	int scale = -1;

	do
	{
		scale++;
		comparator = (BASE_FREQUENCY) / (frequency * pow(2,scale));
		printf("\r PWM2 Comparator: %u\n", comparator);
		printf("\r PWM2 SCALE: %u\n", scale);
	} while (comparator > maxComparator);
	/*
	while(comparator > maxComparator){
		scale += 1;
		comparator = BASE_FREQUENCY / (frequency * pow(2,scale));
		if(comparator <= maxComparator) {
			scale -= 1;
			printf("\r SCALE Break: %u\n", scale);
			break;
		}
		printf("\r SCALE: %u\n", scale);
	}
	*/
	*PWM2_1_CMP0 = 0b0000000000000000;
	*PWM2_1_CMP0 |= comparator;
	*PWM2_1_CONFIG |= scale;
	*PWM2_1_CMP1 = 0b0000000000000000;
	*PWM2_1_CMP1 |= ((comparator * (100 - dutyCycle))/100);
	printf("\r PWM2 Comparator0 Final: %u\n", *PWM2_1_CMP0);
	printf("\r PWM2 Comparator1 Final: %u\n", *PWM2_1_CMP2);
	printf("\r PWM2 SCALE FINAL: %u\n", scale);
	printf("\r PWM2 CONFIG FINAL: %u\n", *PWM2_1_CONFIG);
}

void setSingleShotPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle){
	setPWMPin(pinNumber);
	*PWM0_1_CONFIG = 0b0000000000000000000000000000000;
	*PWM0_1_CONFIG |= (1<<9); 		// ENABLE PWMZeroCMP
	*PWM0_1_CONFIG |= (1<<13); 		// Run PWM01 in ONE-SHOT mode
	printf("\r PWM1 CONFIG FINAL: %u\n", *PWM0_1_CONFIG);
	setPWM1(frequency, dutyCycle);
}

void resetSingleShot() {
	*PWM0_1_CONFIG |= (1<<9); 		// ENABLE PWMZeroCMP
	*PWM0_1_CONFIG |= (1<<13); 		// Run PWM01 in ONE-SHOT mode
}

void setContinuousPWM(uint32_t pinNumber, uint32_t frequency, uint32_t dutyCycle){
	setPWMPin(pinNumber);
	*PWM2_1_CONFIG = 0b0000000000000000000000000000000;
	*PWM2_1_CONFIG |= (1<<9); 		// ENABLE PWMZeroCMP
	*PWM2_1_CONFIG |= (1<<12); 		// Run PWM01 in CONTINUOUS mode
	setPWM2(frequency, dutyCycle);
}

void save_measurement(int time, int angle){
    // uint32_t distance = (speed_of_sound * time)/2;
    measurements[array_element_index].distance = time;
    measurements[array_element_index].angle = angle;
    array_element_index++;
}


void stream_output(int distance, int angle){
    printf("Stream distance results\n");
}


void bubble_sort(struct Measurement *array, int array_size) {

    for(int i = 0; i<array_size; i++) {
        int swaps = 0;         //flag to detect any swap is there or not
        for(int j = 0; j<array_size-i-1; j++) {
            if(array[j].distance > array[j+1].distance) {       //when the current item is bigger than next
                // Swap distance
                //printf("Swap %d with %d\n", array[j].distance, array[j+1].distance);
                int temp;
                temp = array[j].distance;
                array[j].distance = array[j+1].distance;
                array[j+1].distance = temp;

                // Swap angle
                temp = array[j].angle;
                array[j].angle = array[j+1].angle;
                array[j+1].angle = temp;

                swaps = 1;    //set swap flag
            }
        }

        if(!swaps){
            break; // No swap in this pass, so array is sorted
        }
    }
    // Print result
    for(int i=0; i<array_size; i++){
        printf("Array distance sorted[%d]: %d\n", i, array[i].distance);
        printf("Array angle sorted[%d]: %d\n", i, array[i].angle);
    }
}

uint32_t read_echo(){
	uint32_t timeout = get_rtc_low_micro() + sensor_timeout;
	while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1));
	uint32_t startTime = get_rtc_low_micro();
	while(((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1));
	uint32_t stopTime = get_rtc_low_micro();
	if(get_rtc_low_micro() >= timeout){
		printf("\r Sensor Timeout: %u \n", get_rtc_low_micro());
		return(0);
	}
	else {
		printf("\r Got Value\n");
		return(stopTime - startTime);
	}
}


void delay_us(uint32_t delay_in_us){
	int wait_time = get_rtc_low_micro() + delay_in_us;
	while (get_rtc_low_micro() <= wait_time);
}





/*
uint32_t read_echo(uint32_t timeout){
	uint32_t startTime;
	int timeOut = timeout + get_rtc_low_micro();
	while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1) & get_rtc_low_micro() <= timeOut) {
		duration++;
		delayuS(1);
		if(duration>timeout)
		{
			return 0;
		}
	}
	duration=0;

`	while((GPIOA->IDR&GPIO_IDR_ID1)) {
	duration++;
	delayuS(1);
		if(duration>timeout){
			return 0;
		}
	}
	return duration;
}
*/

/*
int main() {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<ECHO_PIN);
	*GPIO_INPUT_EN |= (1<<ECHO_PIN);
	*GPIO_PUE |= (1<<ECHO_PIN);

	while(1) {
		while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1)){
			printf("\r No Input \n");
		}
		printf("\r Reg Val %d \n", *GPIO_INPUT_VAL);
	}
}
*/

////////////////////////////////////////////////////////////////////////////////////////////
// OLD PROGRAMS
////////////////////////////////////////////////////////////////////////////////////////////
/*
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#include "RTC.h"

volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;


int main() {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<ECHO_PIN);
	*GPIO_INPUT_EN |= (1<<ECHO_PIN);
	*GPIO_PUE |= (1<<ECHO_PIN);
	set_bit_high(*GPIO_OUTPUT_EN, TRIG_PIN);	// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)

	rtc_setup();

	while(1){
		*GPIO_OUTPUT_VAL &= ~(1 << TRIG_PIN);													//turn off trig
		int waitTime = get_rtc_low_micro() + 100;
		while(get_rtc_low_micro() <= waitTime);
		*GPIO_OUTPUT_VAL |= (1 << TRIG_PIN);  													//turn on trig
		waitTime = get_rtc_low_micro() + 100;
		while(get_rtc_low_micro() <= waitTime);
		*GPIO_OUTPUT_VAL &= ~(1 << TRIG_PIN);
		while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1));
		uint32_t startTime = get_rtc_low_micro();
		while(((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1));
		uint32_t stopTime = get_rtc_low_micro();
		uint32_t distTime = stopTime - startTime;
		//printf("\r deltaTime: %u \n", distTime);
		uint32_t distance = (distTime * 340) / 2000;
		printf("\r distance mm: %u \n", distance);

		waitTime = get_rtc_low_micro() + 10000;
		while(get_rtc_low_micro() <= waitTime);
	}
}

int main(void) {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<ECHO_PIN);
	*GPIO_INPUT_EN |= (1<<ECHO_PIN);
	*GPIO_PUE |= (1<<ECHO_PIN);
	set_bit_high(*GPIO_OUTPUT_EN, TRIG_PIN);	// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)

	rtc_setup();

	while(1)
	{
		*GPIO_OUTPUT_VAL &= ~(1 << TRIG_PIN);													//turn off trig
		int waitTime = get_rtc_low_micro() + 100;
		while(get_rtc_low_micro() <= waitTime);
		*GPIO_OUTPUT_VAL |= (1 << TRIG_PIN);  													//turn on trig
		waitTime = get_rtc_low_micro() + 10;
		while(get_rtc_low_micro() <= waitTime);
		*GPIO_OUTPUT_VAL &= ~(1 << TRIG_PIN);
		uint32_t startTime = get_rtc_low_micro();
		//duration=read_echo(400000); 								    //measure the time of echo pin
		while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1)) {
			if(get_rtc_low_micro >= startTime + sensor_timeout) {
				distance = 9999;
				break;
			}
		}
		uint32_t stopTime = get_rtc_low_micro();
		printf("\r Distance: %u\n ", distance);
		distance=(stopTime - startTime)/58;											//distance=duration/2*SOUND_SPEED
		printf("\r Distance: %u\n ", distance);
		waitTime = get_rtc_low_micro() + 1000000;
		while(get_rtc_low_micro() <= waitTime);
	}

}

int main() {

	*GPIO_INPUT_EN |= (1 << 9);			// Enable GPIO as input at pin 15 (Echo-Pin)(GPIO 9)
	*GPIO_OUTPUT_EN |= (1 << 11);		// Enable GPIO as input at pin 17 (PWM-Servo)(GPIO 11)
	*GPIO_OUTPUT_EN |= (1 << 1);		// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)

	rtc_setup();

	uint64_t startTime = 0;
	uint64_t stopTime = 0;

	while(1){

		*GPIO_OUTPUT_VAL |= (1 << 1);		// Sets GPIO 1 HIGH
		startTime = get_rtc_low_micro();
		//printf("\r start Time: %u.\n ",startTime);
		//printf("\r Reg: %u. sttime ms: %u\n ",*RTC_OUTPUT_LOW, startTime);

		//printf("\r Reg: %u. ms: %u\n ",*RTC_OUTPUT_LOW, get_rtc_low_milli());
		//printf("\r Reg: %u. us: %u\n ",*RTC_OUTPUT_LOW, get_rtc_low_micro());

		uint64_t tDelay = get_rtc_low_milli() + 10;
		while(!(get_rtc_low_milli() >= tDelay));
		*GPIO_OUTPUT_VAL &= ~(1 << 1);		// Sets GPIO 1 LOW
		uint64_t tWait = get_rtc_low_milli();
		while(!((*GPIO_INPUT_VAL >> 9) & 0b1) & ((get_rtc_low_milli() - tWait) <= 1000));
		stopTime = get_rtc_low_micro();
		//printf("\r Reg: %u. stop Time: %u.\n ",*RTC_OUTPUT_LOW,  stopTime);
		//printf("\r Reg: %u. Stop Time Real: %u.\n ",*RTC_OUTPUT_LOW, get_rtc_low_micro());
		uint64_t timeDiff = stopTime - startTime;
		printf("\r Reg: %u. Start: %u\n ",*RTC_OUTPUT_LOW, startTime);
		printf("\r Reg: %u. Stop: %u\n ",*RTC_OUTPUT_LOW, stopTime);
		printf("\r Reg: %u. TimeDiff: %u.\n ",*RTC_OUTPUT_LOW,  timeDiff);
		while((*GPIO_INPUT_VAL >> 9) & 0b1);

		if(((*GPIO_INPUT_VAL >> 9) & 0b1)){
			stopTime = get_rtc_low_micro();
			uint64_t timeDiff = stopTime - startTime;
			printf("\r TimeDiff: %u.\n ", timeDiff);
		}
		else {
			printf("\r No Time");
		}




		//while(!((*GPIO_INPUT_VAL >> 9) & 0b1) & ((save_rtc_low() - timeStart) <= 100000));

		if(((*GPIO_INPUT_VAL >> 9) & 0b1)){
			int timeStop = save_rtc_low();
			int timeDiff = timeStop - timeStart;
			int length = timeDiff/58;
			printf("\r TimeStop: %d. TimeStart: %d. TimeDiff: %d. Got distance: %d. \n ", timeStop, timeStart, timeDiff, length);
		}



	}
}

*/

/*

int main() {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<ECHO_PIN);
	*GPIO_INPUT_EN |= (1<<ECHO_PIN);
	*GPIO_PUE |= (1<<ECHO_PIN);

	while(1) {
		while(!((*GPIO_INPUT_VAL >> ECHO_PIN) & 0b1)){
			printf("\r No Input \n");
		}
		printf("\r Reg Val %d \n", *GPIO_INPUT_VAL);
	}
}
*/


/*
int main() {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<9);
	*GPIO_INPUT_EN |= (1<<9);
	*GPIO_PUE |= (1<<9);
	//set_bit_high(*GPIO_INPUT_EN, 9); 	// Enable GPIO as input at pin 15 (Echo-Pin)(GPIO 9)
	//set_bit_high(*GPIO_INPUT_EN, 10); 	// Enable GPIO as input at pin 16 (Echo-Pin)(GPIO 9)
	//set_bit_high(*GPIO_INPUT_EN, 11); 	// Enable GPIO as input at pin 17 (PWM-Servo)(GPIO 11)
	set_bit_high(*GPIO_OUTPUT_EN, 1);	// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)
	//set_bit_high(*GPIO_PUE, 9);

	rtc_setup();
	// Initialize servo
	// Move servo to starting position 1 / 65s

	while(1) {
		bool timeout = false;
		int waitTime = get_rtc_low_micro() + 10;
		// Trigger Ultrasound sensor
		set_bit_high(*GPIO_OUTPUT_VAL, 1); 	// Sets pin 1 HIGH
		while(get_rtc_low_micro() <= waitTime);
		set_bit_low(*GPIO_OUTPUT_VAL, 1);	// Sets pin 1 LOW

		// Get starting time
		int start_time = get_rtc_low_micro();
		//printf("\r Start us: %u\n", get_rtc_low_micro());
		// Wait for sensor signal or timeout
		while (((*GPIO_INPUT_VAL >> 9) & 0b1)){
			if (get_rtc_low_micro() >= start_time + sensor_timeout){
				//Signal timeout
				printf("\r TimeOut \n");
				timeout = true;
				break;
			}


		int waitTime = get_rtc_low_micro() + 10;
		// Trigger Ultrasound sensor
		set_bit_high(*GPIO_OUTPUT_VAL, 1); 	// Sets pin 1 HIGH
		while(get_rtc_low_micro() <= waitTime);
		set_bit_low(*GPIO_OUTPUT_VAL, 1);	// Sets pin 1 LOW

		while (((*GPIO_INPUT_VAL >> 9) & 0b1));
		int start_time = get_rtc_low_micro();
		while (!((*GPIO_INPUT_VAL >> 9) & 0b1));
		int end_time = get_rtc_low_micro();

		printf("\r Received signal: %d\n", end_time-start_time);

		int waitTimeResult = get_rtc_low_micro() + 1000000;
		while(get_rtc_low_micro() <= waitTimeResult);

		//printf("\r Reg Val Input %u \n", *GPIO_INPUT_VAL);

		// Only save value if sensor did not timeout

		// Print results in terminal
		// stream_output(int distance, int angle);


		// Move motor to next position here

		// If measurements are over


		Print all array distances:
		for(int i=0; i<array_element_index; i++){
            	printf("Array distance[%d]: %d\n", i, measurements[i].distance);
        	}

		Sort the array:
		bubble_sort(measurements, array_element_index);



		}

		if (!timeout){
			//printf("\r Reg Val %d \n", *GPIO_INPUT_VAL);
			int end_time = get_rtc_low_micro();
			//printf("\r Stop us: %u\n", get_rtc_low_micro());
			printf("\r Received signal: %d, %d\n", end_time-start_time, motor_angle);
			//save_measurement(end_time-start_time, motor_angle);
		}
	}
}
*/


