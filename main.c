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
#define TRIG_PIN 1
#define ECHO_PIN 9

/*FUNCTION DECLARATIONS*/
void save_measurement(int time, int angle);
void stream_output(int distance, int angle);
void bubble_sort(struct Measurement *array, int array_size);
void delay_us(uint32_t delay_in_us);
uint32_t read_echo();



/*Global VARIABLES*/
struct Measurement measurements [ARRAY_MAX_SIZE] = {}; // Up to 65 measurements in the array
uint32_t array_element_index = 0;
uint32_t motor_angle = 0;
uint32_t speed_of_sound = 340; // cm/s
uint32_t sensor_timeout = 60000;  // Units ?
uint32_t loopTime = 80000;


/*PIN DECLARATIONS*/
volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;
volatile uint32_t *GPIO_PUE 		= (uint32_t*)0x10012010;
volatile uint32_t *IOF_EN 			= (uint32_t*)0x10012038;
volatile uint32_t *IOF_EN_SEL 		= (uint32_t*)0x1001203C;

uint32_t read_echo();

void delay_us(uint32_t delay_in_us);


int main() {
	*IOF_EN &= ~(1<<9);
	*GPIO_OUTPUT_EN &= ~(1<<ECHO_PIN);
	*GPIO_INPUT_EN |= (1<<ECHO_PIN);
	*GPIO_PUE |= (1<<ECHO_PIN);
	set_bit_high(*GPIO_OUTPUT_EN, TRIG_PIN);	// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)


	rtc_setup();

	// Initialize servo
	// Move servo to starting position 1 / 65

	while(1) {

		// Trigger sensor

		set_bit_low(*GPIO_OUTPUT_VAL, TRIG_PIN);				// Set pin LOW
		delay_us(10);
		set_bit_high(*GPIO_OUTPUT_VAL, TRIG_PIN); 				// Set pin HIGH
		delay_us(10);
		set_bit_low(*GPIO_OUTPUT_VAL, TRIG_PIN);				// Set pin LOW

		uint32_t distance = (read_echo() * speed_of_sound)/2000;

		printf("\r Distance mm: %u\n", distance);

		while(get_rtc_low_micro() <= loopTime);

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
