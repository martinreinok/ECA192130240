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


/*FUNCTION DECLARATIONS*/
void save_measurement(int time, int angle);
void stream_output(int distance, int angle);
void bubble_sort(struct Measurement *array, int array_size);
void delay_us(uint32_t delay_in_us);
uint32_t read_echo();



/*VARIABLES*/
struct Measurement measurements [ARRAY_MAX_SIZE] = {}; // Up to 65 measurements in the array
int array_element_index = 0;
int motor_angle = 0;
int speed_of_sound = 34300; // cm/s
int sensor_timeout = 1000000;  // Units ?


/*PIN DECLARATIONS*/
volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;


int main() {
	set_bit_high(*GPIO_INPUT_EN, 9); 	// Enable GPIO as input at pin 15 (Echo-Pin)(GPIO 9)
	set_bit_high(*GPIO_INPUT_EN, 11); 	// Enable GPIO as input at pin 17 (PWM-Servo)(GPIO 11)
	set_bit_high(*GPIO_INPUT_EN, 1);	// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)
	rtc_setup();
	// Initialize servo
	// Move servo to starting position 1 / 65

	while(1) {

		// Trigger sensor
		set_bit_low(*GPIO_OUTPUT_VAL, 1);				// Set pin LOW
		delay_us(10);
		set_bit_high(*GPIO_OUTPUT_VAL, 1); 				// Set pin HIGH
		delay_us(10);
		set_bit_low(*GPIO_OUTPUT_VAL, 1);				// Set pin LOW

		// Measure the time until rising edge
		uint32_t distance = read_echo()/58;

		printf("\rObject distance: %d\n", distance);

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



/*FUNCTIONS*/

uint32_t read_echo(){

	uint32_t start_time = get_rtc_low_micro();
	uint32_t timeout = get_rtc_low_micro() + sensor_timeout;

	while (!((*GPIO_INPUT_VAL >> 9) & 0b1)){
		printf("\rWaiting for signal\n");
		if (get_rtc_low_micro() > timeout){
			printf("\rTimeout\n");
			return 0;
		}
	}
	return get_rtc_low_micro() - start_time;
}

void delay_us(uint32_t delay_in_us){
	int wait_time = get_rtc_low_micro() + delay_in_us;
	while (get_rtc_low_micro() <= wait_time) {}
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

