#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "BIT_OPS.h"

volatile uint32_t *RTC_OUTPUT_HIGH 	= (uint32_t*)0x1000004C; // High bits of RTC Counter
volatile uint32_t *RTC_OUTPUT_LOW 	= (uint32_t*)0x10000048; // Low bits of RTC Counter
volatile uint32_t *RTC_CONFIG		= (uint32_t*)0x10000040; // Low bits of RTC Counter

int clock_frequency = 32768;
int clock_multiplier_micro = 100000;

void rtc_setup() {
	set_bit_high(*RTC_CONFIG, 12);
}


uint32_t get_rtc_low() {

	uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	return rtc_low_value;
}

uint32_t get_rtc_low_micro() {

	uint32_t rtc_low_value_micro = *RTC_OUTPUT_LOW;
	return rtc_low_value_micro*clock_multiplier_micro/clock_frequency;
}



uint32_t get_rtc_high() {
	uint32_t rtc_high_value = *RTC_OUTPUT_HIGH;
	return rtc_high_value;
}
