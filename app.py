import streamlit as st


def get_coins():
    with open('data/coins.csv', mode='r') as file:
        coins = [coin.strip() for coin in file.readlines()]
    coins = sorted(coins)
    return coins


def get_timeframes():
    with open('data/timeframes.csv', mode='r') as file:
        timeframes = [tf.strip() for tf in file.readlines()]
    return timeframes


def main():
    with st.sidebar:
        coin = st.selectbox("Coin", get_coins())
        tf = st.selectbox("Timeframe", get_timeframes())
        cnt = st.number_input("Count", min_value=1000, step=100)

    st.title("Hello GeeksForGeeks !!!")

    # Exception - This has been added later
    exp = ZeroDivisionError("Trying to divide by Zero")
    st.exception(exp)

    # Write text
    st.write("Привет медведь")

    if st.checkbox("Show/Hide"):
        st.text("Showing the widget")
    else:
        st.text("Hiding the widget")

    hobby = st.selectbox("Hobbies: ",
                         ['Dancing', 'Reading', 'Sports'])
    st.write("Your hobby is: ", hobby)

    st.button("Click me for no reason")
    if st.button("About"):
        st.text("Welcome To GeeksForGeeks!!!")

    name = st.text_input("Enter Your name", "Type Here ...")
    if st.button('Submit'):
        result = name.title()
        st.success(result)


if __name__ == '__main__':
    main()
