import re
import unicodedata

from bs4 import BeautifulSoup, PageElement


class HTMLExtractor:
    """
    A class to extract and process various elements from HTML content
    using BeautifulSoup.

    Attributes:
        soup (BeautifulSoup): A BeautifulSoup object.
    """

    def __init__(self, html_content: str):
        """
        Initializes the HTMLExtractor with the given HTML content.

        Args:
            html_content (str): The HTML content to be processed.
        """
        self.soup = self.preprocess_html(html_content)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans the given text by normalizing Unicode characters,
        handling special symbols, replacing problematic characters,
        and removing multiple whitespace.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Normalize Unicode characters
        text = unicodedata.normalize("NFKD", text)
        # Use ASCII encoding to handle special symbols e.g. copyright \xa9
        text = text.encode("ascii", "ignore").decode("utf-8")

        # Replace common problematic characters
        text = text.replace("\xa0", " ")  # non-breaking space
        text = text.replace("\u200b", "")  # zero-width space
        text = text.replace("\u2028", "\n")  # line separator
        text = text.replace("\u2029", "\n")  # paragraph separator

        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def preprocess_html(cls, html_content: str) -> BeautifulSoup:
        """
        Preprocesses the given HTML content by replacing all <br>
        tags with newline characters.

        Args:
            html_content (str): The HTML content to be preprocessed.

        Returns:
            BeautifulSoup: The preprocessed HTML content as a BeautifulSoup object.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all <br> tags and replace them with newline
        for br in soup.find_all("br"):
            br.replace_with("\n")

        return soup

    def check_for_table(self) -> bool:
        """
        Check for the presence of table tags in an HTML document.

        Returns:
            bool: True if at least one table tag is found, False otherwise.
        """
        # Find all table tags
        tables = self.soup.find_all("table")

        # Return True if at least one table is found, False otherwise
        return len(tables) > 0

    def check_for_image(self) -> bool:
        """
        Check for the presence of img tags in an HTML document.

        Returns:
            bool: True if at least one img tag is found, False otherwise.
        """
        # Find all img tags
        images = self.soup.find_all("img")

        # Return True if at least one image is found, False otherwise
        return len(images) > 0

    def extract_related_sections(self) -> list[str]:
        """
        Extracts "Related:" sections and "Read these next:" items from the HTML content.

        Returns:
            list[str]: A list of related sections and "Read these next:" items.
        """
        related_sections = []
        read_these_next_ul = None
        # Extract "Related:" sections and "Read these next:" items
        for tag in self.soup.find_all(["p", "ul"]):
            if tag.name == "p" and tag.find("strong"):
                if "Related:" in tag.text:
                    related_sections.append(
                        re.sub(r"Related: ", "", self.clean_text(tag.text))
                    )
                elif "Read these next:" in tag.text:
                    read_these_next_ul = tag.find_next_sibling("ul")
            elif tag == read_these_next_ul:
                for li in tag.find_all("li"):
                    related_sections.append(self.clean_text(li.text))

        return related_sections

    def extract_text(self) -> str:
        """
        Extracts the main content from the HTML content.

        Returns:
            str: The main content body extracted from the HTML content.

        Note:
            This function unwraps the HTML content if it is contained in a <div>. It then extracts the
            main content by iterating over the tags in the soup. The following tags are considered:

                - h1, h2, h3, h4, h5, h6: These tags are treated as key headers and are paragraphed between them.
                - p: This tag is treated as a paragraph. <em> tags are removed from the text.
                    * If the text does not contain sentences about HealthHub app, Google Play, or Apple Store,
                    and it contains a strong tag, it is treated differently based on the text content.
                - ul: This tag is treated as an unordered list. If it is the child of a <div>, it is treated as a list.
                - ol: This tag is treated as an ordered list.
                - div: This tag is treated as a text within a div.
                - span: This tag is treated as a text within a span.

            The extracted content is stored in a list and then processed. Double newlines are replaced with single
            newlines and whitespace is stripped. If the processed text is empty, the function attempts to extract the
            content from the <div> tags.
        """
        # Unwrap if the HTML content is contained in a div
        if self.soup.div is not None:
            self.soup.div.unwrap()

        # Extract the main content
        content = []
        for tag in self.soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "div", "span"],
            recursive=False,
        ):
            if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # Provide paragraphing between key headers
                content.append("\n")
                content.append(self.clean_text(tag.text))

            elif tag.name == "p":
                # Remove all em tags
                for em in tag.find_all("em"):
                    em.extract()
                # Get the remaining text
                text = tag.get_text()
                # Remove sentences about HealthHub app, Google Play, and Apple Store
                if not re.search(
                    r"(HealthHub app|Google Play|Apple Store|Parent Hub)", text
                ):
                    if tag.find("strong"):
                        if "Related:" in tag.text:
                            text = self.clean_text(tag.text)
                            content.append(re.sub(r"\n", " ", text))
                        elif "Read these next:" in tag.text:
                            content.append(self.clean_text(tag.text))
                    else:
                        content.append(self.clean_text(text))

            # For unordered lists
            elif tag.name == "ul":
                # not "ul" so we avoid duplicates
                for li in tag.find_all("li"):
                    content.append("- " + self.clean_text(li.text))

            # For ordered lists
            elif tag.name == "ol":
                for i, li in enumerate(tag.find_all("li")):
                    content.append(f"{i + 1}. " + self.clean_text(li.text))

            # For texts within div
            elif tag.name in ["div", "span"]:
                # Changes content via Pass By Reference
                self._extract_text_from_container(tag, content)

            content.append("")  # Add a blank line after each element

        # Remove empty strings from content
        content = [c for c in content if c]

        # Replace double newlines with single newlines and strip whitespace
        extracted_content_body = "\n".join(content).replace("\n\n", "\n").strip()

        return extracted_content_body

    def _extract_text_from_container(
        self, tag: PageElement, content: list[str]
    ) -> None:
        """
        Helper method to extract text from div and span containers.

        This method recursively processes div and span elements, extracting their text content
        while avoiding duplication. It handles both direct text content and nested elements.

        Args:
            tag (PageElement): The BeautifulSoup tag (div or span) to extract text from.
            content (list[str]): The list to append extracted text to.

        Returns:
            None: This method modifies the content list in-place.
        """

        if tag.text:
            content.append("")
            text_fragment = self.clean_text(tag.text)
            if len(content) > 0 and content[-1] != text_fragment:
                content.append(text_fragment)
        else:
            for child in tag.children:
                if child.name not in ["div", "span", "ul", "ol"]:
                    content.append("")
                    text_fragment = self.clean_text(child.text)
                    if len(content) > 0 and content[-1] != text_fragment:
                        content.append(text_fragment)
                elif child.name in ["div", "span"]:
                    self._extract_text_from_container(child, content)

    def extract_links(self) -> list[tuple[str, str]]:
        """
        Extracts the title and URL from all the anchor tags in the HTML content.

        Returns:
            list[tuple[str, str]]:
                A list of tuples containing the title and URL of each anchor tag.

        Note:
            Footnotes to references sections are ignored.
        """
        extracted_links = []

        # Extract title/text and links from anchor tags
        for link in self.soup.find_all("a"):
            url = link.get("href")
            # Ignore footnotes
            if url != "#footnotes":
                text = link.get("title") or link.get_text()
                record = text, url
                extracted_links.append(record)

        return extracted_links

    def extract_headers(self) -> list[tuple[str, str]]:
        """
        Extracts the headers from the HTML content.

        Returns:
            list[tuple[str, str]]:
                A list of tuples containing the text and tag name of
                each header found in the HTML content.
        """
        extracted_headers = []

        for title in self.soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            tag = title.name
            text = title.get_text()
            record = text, tag
            extracted_headers.append(record)

        return extracted_headers
