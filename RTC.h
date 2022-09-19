#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "BIT_OPS.h"

volatile uint32_t *RTC_OUTPUT_HIGH 	= (uint32_t*)0x1000004C; // High bits of RTC Counter
volatile uint32_t *RTC_OUTPUT_LOW 	= (uint32_t*)0x10000048; // Low bits of RTC Counter
volatile uint32_t *RTC_CONFIG		= (uint32_t*)0x10000040; // Low bits of RTC Counter

const uint64_t clock_frequency = 32768;
const uint64_t clock_multiplier_micro = 1000000;
const uint64_t clock_multiplier_milli = 1000;

void rtc_setup() {
	set_bit_low(*RTC_CONFIG, 0);
	set_bit_low(*RTC_CONFIG, 1);
	set_bit_low(*RTC_CONFIG, 2);

	set_bit_high(*RTC_CONFIG, 12);
}


uint32_t get_rtc_low() {
	volatile uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	return rtc_low_value;
}

uint32_t get_rtc_low_micro() {

	volatile uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	uint64_t rtc_low_value_Multi = rtc_low_value * clock_multiplier_micro;
	uint32_t result = rtc_low_value_Multi / clock_frequency;
	return (result);
}

uint32_t get_rtc_low_milli() {

	volatile uint32_t rtc_low_value = *RTC_OUTPUT_LOW;
	return ((rtc_low_value * clock_multiplier_milli)/clock_frequency);
}

uint32_t get_rtc_high() {
	volatile uint32_t rtc_high_value = *RTC_OUTPUT_HIGH;
	return rtc_high_value;
}
