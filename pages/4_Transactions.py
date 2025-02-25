import streamlit as st

from utils.streamlit_utils import (
    COLUMN_CONFIG,
    download_transactions,
    read_data,
    render_page,
    upload_data,
)


def render_transaction_tab():
    def change_transactions():
        st.session_state["transactions"] = st.session_state["edited_transactions"]

    transactions = st.session_state.get("transactions")
    st.header("Current Transactions")

    if transactions is not None:
        st.markdown("You can **Add** | **Edit** | **Remove** the transactions:")
        st.session_state["edited_transactions"] = transactions.copy()
        edited_transactions = st.data_editor(
            st.session_state["edited_transactions"],
            use_container_width=True,
            num_rows="dynamic",
            column_config=COLUMN_CONFIG,
        )
        st.session_state["edited_transactions"] = edited_transactions
        st.button("Submit changes", on_click=change_transactions)
        download_transactions(edited_transactions)


if __name__ == "__main__":
    result = render_page() if render_page else None

    if result is not None:
        render_transaction_tab()
