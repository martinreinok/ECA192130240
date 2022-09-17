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

	while(1){

		//*GPIO_OUTPUT_VAL |= (1 << 1);		// Sets GPIO 1 HIGH
		//startTime = get_rtc_low_micro();
		//printf("\r start Time: %u.\n ",startTime);
		printf("\r start Time: %u.\n ",get_rtc_low_milli());
		printf("\r start Time: %u.\n ",*RTC_OUTPUT_LOW);
		/*
		uint64_t tDelay = get_rtc_low_milli() + 10;
		while(!(get_rtc_low_milli() >= tDelay));

		*GPIO_OUTPUT_VAL &= ~(1 << 1);		// Sets GPIO 1 LOW
		uint64_t tWait = get_rtc_low_milli();
		while(!((*GPIO_INPUT_VAL >> 9) & 0b1) & ((get_rtc_low_milli() - tWait) <= 10000));
		stopTime = get_rtc_low_micro();
		printf("\r stop Time: %u.\n ", stopTime);
		printf("\r Stop Time Real: %u.\n ",get_rtc_low_micro());
		uint64_t timeDiff = stopTime - startTime;
		printf("\r TimeDiff: %u.\n ", timeDiff);
		*/
		/*
		if(((*GPIO_INPUT_VAL >> 9) & 0b1)){
			stopTime = get_rtc_low_micro();
			uint64_t timeDiff = stopTime - startTime;
			printf("\r TimeDiff: %u.\n ", timeDiff);
		}
		else {
			printf("\r No Time");
		}
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
