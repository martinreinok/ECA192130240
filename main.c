#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "RTC.h"

//custom write delay function since we do not have one like an Arduino
void delay(int number_of_microseconds){ //not actually number of seconds

// Converting time into multiples of a hundred nS
int hundred_ns = 10 * number_of_microseconds;

// Storing start time
clock_t start_time = clock();

// looping till required time is not achieved
while (clock() < start_time + hundred_ns);

}

volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;

int main() {

	*GPIO_INPUT_EN |= (1 << 9);			// Enable GPIO as input at pin 15 (Echo-Pin)(GPIO 9)
	*GPIO_OUTPUT_EN |= (1 << 11);		// Enable GPIO as input at pin 17 (PWM-Servo)(GPIO 11)
	*GPIO_OUTPUT_EN |= (1 << 1);		// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)
	rtc_setup();

	while(1){

		*GPIO_OUTPUT_VAL |= (1 << 1);		// Sets pin 1 HIGH

		*GPIO_OUTPUT_VAL &= ~(1 << 1);		// Sets pin 1 LOW

		printf("\r Clock value: %d \n", get_rtc_low_micro());
		/*
		while(!((*GPIO_INPUT_VAL >> 9) & 0b1));
		int timeStart = save_rtc_low();
		while(!((*GPIO_INPUT_VAL >> 9) & 0b0) & ((save_rtc_low() - timeStart) <= 100000));
		int timeStop = save_rtc_low();
		int timeDiff = timeStop - timeStart;
		printf("\r TimeStart: %d. TimeStop: %d. TimeDiff: %d.\n ", timeStart/32000, timeStop/32000, timeDiff);
		*/
		//while(!((*GPIO_INPUT_VAL >> 9) & 0b1) & ((save_rtc_low() - timeStart) <= 100000));
		/*
		if(((*GPIO_INPUT_VAL >> 9) & 0b1)){
			int timeStop = save_rtc_low();
			int timeDiff = timeStop - timeStart;
			int length = timeDiff/58;
			printf("\r TimeStop: %d. TimeStart: %d. TimeDiff: %d. Got distance: %d. \n ", timeStop, timeStart, timeDiff, length);
		}
		*/
	}
}
