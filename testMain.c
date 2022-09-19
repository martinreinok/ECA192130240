#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include "BIT_OPS.h"
#include "RTC.h"

/*DATA STRUCT*/
struct Measurement{
    int distance;
    int angle;
};


/*GLOBAL DEFINES*/
#define ARRAY_MAX_SIZE 65


/*GLOBAL VARIABLES*/
struct Measurement measurements [ARRAY_MAX_SIZE] = {}; // Up to 65 measurements in the array
int array_element_index = 0;
int motor_angle = 0;
const int speed_of_sound = 34300; // cm/s
const int sensor_timeout = 1000000;  // Units ?


/*PIN DECLARATIONS*/
volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;
volatile uint32_t *GPIO_PUE 		= (uint32_t*)0x10012010;
volatile uint32_t *IOF_EN 			= (uint32_t*)0x10012038;
volatile uint32_t *IOF_EN_SEL 		= (uint32_t*)0x1001203C;

/*FUNCTION DECLARATIONS*/
void save_measurement(int time, int angle);
void stream_output(int distance, int angle);
void bubble_sort(struct Measurement *array, int array_size);


/*
int main() {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<9);
	*GPIO_INPUT_EN |= (1<<9);
	*GPIO_PUE |= (1<<9);

	while(1) {
		while(!((*GPIO_INPUT_VAL >> 9) & 0b1)){
			printf("\r No Input \n");
		}
		printf("\r Reg Val %d \n", *GPIO_INPUT_VAL);
	}
}
*/

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
		/*
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
		 */


		}
	/*
		if (!timeout){
			//printf("\r Reg Val %d \n", *GPIO_INPUT_VAL);
			int end_time = get_rtc_low_micro();
			//printf("\r Stop us: %u\n", get_rtc_low_micro());
			printf("\r Received signal: %d, %d\n", end_time-start_time, motor_angle);
			//save_measurement(end_time-start_time, motor_angle);
		}
	}*/
}



/*FUNCTIONS*/

void save_measurement(int time, int angle){
    // uint32_t distance = (speed_of_sound * time)/2;
    measurements[array_element_index].distance = time;
    measurements[array_element_index].angle = angle;
    array_element_index++;
}


void stream_output(int distance, int angle){
    printf("\r Stream distance results\n");
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
        printf("\r Array distance sorted[%d]: %d\n", i, array[i].distance);
        printf("\r Array angle sorted[%d]: %d\n", i, array[i].angle);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
// OLD PROGRAM
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
