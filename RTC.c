volatile uint32_t *RTC_OUTPUT_HIGH 	= (uint32_t*)0x1000004C; // High bits of RTC Counter
volatile uint32_t *RTC_OUTPUT_LOW 	= (uint32_t*)0x10000048; // Low bits of RTC Counter
volatile uint32_t *RTC_CONFIG		= (uint32_t*)0x10000040; // Low bits of RTC Counter

int save_rtc_low() {
	int rtc_low_value = *RTC_OUTPUT_LOW;
	retrun rtc_low_value;
}

int save_rtc_high() {
	int rtc_high_value = *RTC_OUTPUT_HIGH;
	retrun rtc_high_value;
}